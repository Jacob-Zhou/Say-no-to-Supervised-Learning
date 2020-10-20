# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM,
                           Triaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRF2oDependency, CRFDependency, MatrixTree
from supar.utils import Config
from supar.utils.alg import differentiable_eisner, eisner, eisner2o, mst, crf
from supar.utils.transform import CoNLL
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.nn.functional import one_hot, log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineDependencyModel(nn.Module):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_feats (int):
            Size of the feat vocabulary.
        n_rels (int):
            Number of labels in the treebank.
        feat (str):
            Specifies which type of additional feature to use: 'char' | 'bert' | 'tag'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained langugae models like `XLNet` are also feasible.
            'tag': POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_feat_embed (int):
            Size of feature representations. Default: 100.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use. Required if feat='bert'.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float): Default: .33.
            Dropout ratio of LSTM.
        n_mlp_arc (int):
            Arc MLP size. Default: 500.
        n_mlp_rel  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_rels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed+n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats):
        """
        Args:
            words (torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
        """

        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.

        Returns:
            arc_preds (torch.Tensor): [batch_size, seq_len]
                The predicted arcs.
            rel_preds (torch.Tensor): [batch_size, seq_len]
                The predicted labels.
        """

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class CRFNPDependencyModel(BiaffineDependencyModel):
    """
    The implementation of non-projective CRF Dependency Parser.

    References:
        - Xuezhe Ma and Eduard Hovy (IJCNLP'17)
          Neural Probabilistic Model for Non-projective MST Parsing
          https://www.aclweb.org/anthology/I17-1007/
        - Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins (ACL'07)
          Structured Prediction Models via the Matrix-Tree Theorem
          https://www.aclweb.org/anthology/D07-1015/
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.matrix_tree = MatrixTree()

    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=True):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
            arc_probs (torch.Tensor): [batch_size, seq_len, seq_len]
                Orginal arc scores if mbr is False, marginals otherwise.
        """

        batch_size, seq_len = mask.shape
        arc_loss, arc_probs = self.matrix_tree(s_arc, mask, arcs, mbr)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs


class VAEDependencyModel(BiaffineDependencyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_embed       = kwargs['n_embed']
        n_lstm_hidden = kwargs['n_lstm_hidden']
        n_mlp_dec     = kwargs['n_mlp_dec']
        n_words       = kwargs['n_words']

        self.fc_mu = nn.Linear(n_lstm_hidden*2, n_mlp_dec)
        self.fc_var = nn.Linear(n_lstm_hidden*2, n_mlp_dec)

        self.mlp_gnn = MLP(n_mlp_dec, n_mlp_dec*2)
        # self.mlp_gnn = MLP(n_mlp_dec, n_mlp_dec)

        self.mlp_dec = MLP(n_mlp_dec, n_embed, activation=nn.Identity())

        self.generator = nn.Linear(n_embed, n_words, bias=False)
        # self.generator.weight = self.word_embed.weight

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats, supervised_mask=None, arcs=None):
        """
        Args:
            words (torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)


        if not self.training:
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
            return s_arc, s_rel

        # [batch_size, seq_len, hidden_size*2]
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        std = torch.exp(0.5 * log_var)

        # Reparameterization trick for z
        eps = torch.randn_like(std)
        z = eps * std + mu

        tree = torch.zeros_like(s_arc)

        # shift left predict
        # [batch_size, seq_len]
        mask2d = mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1)
        mask2d.diagonal(dim1=-2, dim2=-1)[:] = 0

        if supervised_mask is None or arcs is None:
            supervised_mask = mask.new_zeros([batch_size])

        ent = None

        if supervised_mask.any():
            # [supervised_batch_size, seq_len]
            gold_tree = arcs[supervised_mask]
            # [supervised_batch_size, seq_len, seq_len]
            gold_tree = one_hot(gold_tree, num_classes=seq_len)
            tree[supervised_mask] = gold_tree.float()

        eps = torch.finfo(torch.float).eps

        if not supervised_mask.all():
            # [unsupervised_batch_size, seq_len, seq_len]
            s_unsuper_arc = s_arc[~supervised_mask]
            uni_dist = Uniform(eps, 1-eps)
            noise = uni_dist.sample(s_unsuper_arc.shape)
            noise = -(-noise.log()).log().to(s_unsuper_arc)
            s_tree = s_unsuper_arc + noise
            s_sample_tree = torch.masked_fill(s_tree, ~mask2d[~supervised_mask], float('-inf'))
            sample_tree = s_sample_tree.softmax(-1)
            ent = sample_tree * s_sample_tree.log_softmax(-1).masked_fill(~mask2d[~supervised_mask], 0)
            tree[~supervised_mask] = sample_tree.float()
        # if torch.isnan(ent).any():
        #     print()
        #     # heatmap(sample_tree.mean(0), 'p')
        #     print(sample_tree[torch.isnan(ent)])
        #     print(noise[torch.isnan(ent)])
        #     heatmap(ent.mean(0))
        #     print(sample_tree.shape)

        r_head, r_dep = self.mlp_gnn(z).chunk(2, dim=-1)

        r_head = torch.einsum("bxd,bxy->byd", r_head, tree) / (tree.sum(1)[..., None]+eps)
        r_dep  = torch.einsum("bxd,byx->byd", r_dep, tree) / (tree.sum(2)[..., None]+eps)

        r_dec  = self.mlp_dec(r_head + r_dep)
        s_word = self.generator(r_dec)

        # [batch_size, seq_len, hidden_size*2]
        z_kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1))
        y_kld_loss = -ent.sum(-1).mean() if ent is not None else 0.
        return s_arc, s_rel, s_word, (z_kld_loss, y_kld_loss)

    def loss(self, s_arc, s_rel, s_word, arcs, rels, words, kld_loss, mask, word_mask, supervised_mask=None):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len]
                Mask for covering the unpadded tokens.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
        """
        batch_size, _ = mask.shape
        if supervised_mask is None:
            supervised_mask = mask.new_ones(batch_size)

        supervised_mask = mask & supervised_mask[:, None]

        # supervised loss
        arc_loss = 0.
        rel_loss = 0.
        if supervised_mask.any():
            s_arc, arcs = s_arc[supervised_mask], arcs[supervised_mask]
            s_rel, rels = s_rel[supervised_mask], rels[supervised_mask]
            s_rel = s_rel[torch.arange(len(arcs)), arcs]
            arc_loss = self.criterion(s_arc, arcs)
            rel_loss = self.criterion(s_rel, rels)

        # reconstruction loss (no matter supervised learning or unsupervised learning)
        s_word, words = s_word[word_mask], words[word_mask]
        recons_loss = self.criterion(s_word, words)
        z_kld_loss, y_kld_loss = kld_loss
        # print()
        # print(f"arc_loss:    {arc_loss :.3f}")
        # print(f"rel_loss:    {rel_loss :.3f}")
        # print(f"recons_loss: {recons_loss :.3f}")
        # print(f"z_kld_loss:  {z_kld_loss :.3f}")
        # print(f"y_kld_loss:  {y_kld_loss :.3f}")
        # if torch.isnan(arc_loss + rel_loss + recons_loss + z_kld_loss + y_kld_loss):
        #     exit()

        return self.args.tree_weight * (arc_loss + rel_loss) + \
            self.args.recons_weight * recons_loss + \
            self.args.z_kld_weight * z_kld_loss + \
            self.args.y_kld_weight * y_kld_loss

class CRFDependencyModel(BiaffineDependencyModel):
    """
    The implementation of first-order CRF Dependency Parser.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
          Efficient Second-Order TreeCRF for Neural Dependency Parsing
          https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.crf = CRFDependency()

    def loss(self, s_arc, s_rel, arcs, rels, mask, supervised_mask=None, mbr=True, partial=False, unsuper_loss=False):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len]
                Mask for covering the unpadded tokens.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.
            partial (bool):
                True denotes the trees are partially annotated. Default: False.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
            arc_probs (torch.Tensor): [batch_size, seq_len, seq_len]
                Orginal arc scores if mbr is False, marginals otherwise.
        """

        batch_size, seq_len = mask.shape
        if supervised_mask is None:
            supervised_mask = mask.new_ones(batch_size)
        arc_loss, arc_probs = self.crf(s_arc, mask, supervised_mask, arcs, mbr, partial, unsuper_loss)
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        if not supervised_mask.all():
            mask = mask & supervised_mask.unsqueeze(-1)
        if mask.any():
            s_rel, rels = s_rel[mask], rels[mask]
            s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
            rel_loss = self.criterion(s_rel, rels)
        else:
            rel_loss = 0
        loss = arc_loss + rel_loss
        return loss, arc_probs


class CRF2oDependencyModel(BiaffineDependencyModel):
    """
    The implementation of second-order CRF Dependency Parser.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
          Efficient Second-Order TreeCRF for Neural Dependency Parsing
          https://www.aclweb.org/anthology/2020.acl-main.302/

    Args:
        Remainings required arguments are listed in BiaffineDependencyModel.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        lstm_dropout (float):
            Dropout ratio of LSTM. Default: .33.
        n_mlp_sib (int):
            Sibling MLP size. Default: 500.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
    """

    def __init__(self, n_lstm_hidden=400, n_mlp_sib=100, mlp_dropout=.33, **kwargs):
        super().__init__(**kwargs)

        self.mlp_sib_s = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_sib,
                             dropout=mlp_dropout)
        self.mlp_sib_d = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_sib,
                             dropout=mlp_dropout)
        self.mlp_sib_h = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_sib,
                             dropout=mlp_dropout)

        self.sib_attn = Triaffine(n_in=n_mlp_sib,
                                  bias_x=True,
                                  bias_y=True)
        self.crf = CRF2oDependency()

    def forward(self, words, feats):
        """
        Args:
            words (torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_sib (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
                The scores of all possible dependent-head-sibling triples.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        sib_s = self.mlp_sib_s(x)
        sib_d = self.mlp_sib_d(x)
        sib_h = self.mlp_sib_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask, mbr=True, partial=False):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_sib (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
                The scores of all possible dependent-head-sibling triples.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            sibs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard siblings.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.
            partial (bool):
                True denotes the trees are partially annotated. Default: False.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
            arc_probs (torch.Tensor): [batch_size, seq_len, seq_len]
                Orginal arc scores if mbr is False, marginals otherwise.
        """

        batch_size, seq_len = mask.shape
        scores, target = (s_arc, s_sib), (arcs, sibs)
        arc_loss, arc_probs = self.crf(scores, mask, target, mbr, partial)
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs

    def decode(self, s_arc, s_sib, s_rel, mask, tree=False, mbr=True, proj=False):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_sib (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
                The scores of all possible dependent-head-sibling triples.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            mbr (bool):
                If True, performs MBR decoding. Default: True.
            proj (bool):
                If True, ensures to output projective trees. Default: False.

        Returns:
            arc_preds (torch.Tensor): [batch_size, seq_len]
                The predicted arcs.
            rel_preds (torch.Tensor): [batch_size, seq_len]
                The predicted labels.
        """

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            if proj and not mbr:
                arc_preds = eisner2o((s_arc, s_sib), mask)
            else:
                alg = eisner if proj else mst
                arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


def heatmap(corr, name='matrix'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 4))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    cmap = "RdBu"

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr.detach().cpu(), cmap=cmap, center=0, ax=ax,
                # vmin=-1.1, vmax=1.1, 
                square=True, linewidths=.5,
                xticklabels=False, yticklabels=False,
                cbar=False)
    plt.savefig(f'{name}.png')
    plt.close()