# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (MLP, BertEmbedding, Biaffine, CharLSTM,
                           Triaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRF2oDependency, CRFDependency, MatrixTree
from supar.utils import Config
from supar.utils.alg import eisner, eisner2o, mst, kmeans
from supar.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.functional import one_hot
from supar.utils.fn import heatmap, reverse_padded_sequence

class POSModel(nn.Module):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_cpos (int):
            Size of the pos vocabulary.
        normalize_paras (bool):
            TODO
    """

    def __init__(self,
                 n_words,
                 n_cpos,
                 normalize_paras=False,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        # emit prob
        self.E = nn.Parameter(torch.ones(n_words, n_cpos))

        # transfer prob
        self.T = nn.Parameter(torch.zeros(n_cpos, n_cpos))
        self.start = nn.Parameter(torch.zeros(n_cpos))
        self.end = nn.Parameter(torch.zeros(n_cpos))

        self._params = nn.ParameterDict({
                'E': self.E,
                'T': self.T,
                'start': self.start,
                'end': self.end,
        })

        self.eps = 1e-6
        self.gamma_sum  = 0
        self.start_sum  = 0
        self.end_sum    = 0
        self.emit_sum   = 0
        self.xi_sum     = 0
        self.total_sent = 0
        self.reset_parameters()

    def reset_parameters(self):
        if self.args.em_alg:
            nn.init.normal_(self.E.data)
            self.E.data        = self.E.data.log_softmax(0)
            self.T.data        = self.T.log_softmax(-1)
            self.start.data    = self.start.data.log_softmax(-1)
            self.end.data      = self.end.data.log_softmax(-1)
        else:
            nn.init.normal_(self.E.data, 0, 3)
            nn.init.normal_(self.T.data, 0, 3)
            nn.init.normal_(self.start.data, 0, 3)
            nn.init.normal_(self.end.data, 0, 3)

    def load_pretrained(self, embed=None, smoothing=0.01):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed/(2*self.n_feats**0.5))
            nn.init.normal_(self.embedding.weight.data, 0, 1/(2*self.n_feats**0.5))
            nn.init.zeros_(self.embedding.weight.data)

        return self

    def forward(self, words, mask):
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
        # [n_words, n_cpos]
        emit_probs = self.E.log_softmax(0) if self.args.normalize_paras else self.E
        # [batch_size, seq_len, n_cpos]
        emit_probs = nn.functional.embedding(words, emit_probs)
        emit_probs[~mask] = float('-inf')
        return emit_probs, self.T.log_softmax(-1) if self.args.normalize_paras else self.T

    def baum_welch(self, words, mask, emit_probs, trans_probs):
        gamma, xi, logP = self._e_step(emit_probs, trans_probs, mask)
        self._m_step(words, mask, gamma, xi)
        return logP
    
    def get_logP(self, emit_probs, trans_probs, mask):
        return self._light_forward(emit_probs, trans_probs, mask)

    def _light_forward(self, emit_probs, trans_probs, mask):
        # the end position of each sentence in a batch
        seq_len = emit_probs.shape[1]
        emit_probs = emit_probs.permute(1, 0, 2).double()
        mask       = mask.permute(1, 0)
        start_probs = self.start.log_softmax(-1) if self.args.normalize_paras else self.start
        end_probs = self.end.log_softmax(-1) if self.args.normalize_paras else self.end
        # alpha: [batch_size, seq_len, n_cpos]
        alpha = start_probs.unsqueeze(0) + emit_probs[0]

        for i in range(1, seq_len):
            # [masked_batch_size, n_cpos_pre, n_cpos_now] -> [masked_batch_size, n_cpos_now]
            scores = alpha.unsqueeze(-1) + \
                        emit_probs[i].unsqueeze(1) + \
                        trans_probs

            scores = torch.logsumexp(scores[mask[i]], dim=1)
            alpha[mask[i]] = scores

        last_alphabeta = alpha + end_probs.unsqueeze(0)

        logP = torch.logsumexp(last_alphabeta, dim=-1)

        return logP

    def _forward(self, emit_probs, trans_probs, mask, forward=True):
        # the end position of each sentence in a batch
        lens = mask.sum(-1)
        emit_probs = emit_probs.double()
        batch_size, seq_len, n_cpos = emit_probs.shape
        # alphabeta/pointer: [batch_size, seq_len, n_cpos]
        alphabeta = emit_probs.new_zeros(batch_size, seq_len, n_cpos, dtype=torch.double).log()
        pointer = emit_probs.new_zeros(batch_size, dtype=torch.long) if forward else (lens-1)
        if forward:
            alphabeta[torch.arange(batch_size), pointer] = self.start.unsqueeze(0) + emit_probs[:, 0]
        else:
            alphabeta[torch.arange(batch_size), pointer] = self.end.unsqueeze(0).double()
        pointer += (1 if forward else -1)
        while True:
            compute_mask = (pointer < lens) if forward else (pointer >= 0)
            if not compute_mask.any():
                break
            masked_batch_idx = torch.arange(batch_size)[compute_mask]
            masked_pointer   = pointer[compute_mask]
            if forward:
                # [masked_batch_size, n_cpos_pre, n_cpos_now] -> [masked_batch_size, n_cpos_now]
                scores = alphabeta[masked_batch_idx, masked_pointer-1].unsqueeze(-1) + \
                            emit_probs[masked_batch_idx, masked_pointer].unsqueeze(1) + \
                            trans_probs
            else:
                # [masked_batch_size, n_cpos_now, n_cpos_next] -> [masked_batch_size, n_cpos_now]
                scores = alphabeta[masked_batch_idx, masked_pointer+1].unsqueeze(1) + \
                            emit_probs[masked_batch_idx, masked_pointer+1].unsqueeze(1) + \
                            trans_probs
                scores = scores.permute(0, 2, 1)
            scores = torch.logsumexp(scores, dim=1)
            alphabeta[masked_batch_idx, masked_pointer] = scores
            pointer += (1 if forward else -1)

        last_alphabeta = alphabeta[torch.arange(batch_size), lens-1] if forward else alphabeta[:, 0]
        last_alphabeta = last_alphabeta + (self.end.unsqueeze(0) if forward else (self.start.unsqueeze(0) + emit_probs[:, 0]))

        logP = torch.logsumexp(last_alphabeta, dim=-1)

        return alphabeta, logP

    def decode(self, emit_probs, trans_probs, mask):
        lens = mask.sum(-1)
        mask = mask.t()
        batch_size, seq_len, n_cpos = emit_probs.shape
        p     = emit_probs.new_zeros(seq_len, batch_size, n_cpos).long()
        alpha = emit_probs.new_zeros(batch_size, n_cpos)
        start_probs = self.start.log_softmax(-1) if self.args.normalize_paras else self.start
        alpha[:] = start_probs.unsqueeze(0) + emit_probs[:, 0]

        for i in range(1, seq_len):
            # [batch_size, n_labels_pre, 1] + [batch_size, n_labels_pre, n_label_now]
            _s = alpha.unsqueeze(-1) + \
                    emit_probs[:, i].unsqueeze(1) + \
                    trans_probs
            _s, _p = _s.max(1)
            alpha[mask[i]] = _s[mask[i]]
            p[i, mask[i]] = _p[mask[i]]
        _, p_end = alpha.max(1)

        def backtrack(path, l, now_label):
            labels = [now_label]
            for i in range(l, 0, -1):
                this = path[i][now_label]
                labels.append(this)
                now_label = this
            return torch.tensor(list(reversed(labels)))

        p = p.permute(1, 0, 2).tolist()
        p_end = p_end.tolist()
        sequences = [backtrack(p[i], length-1, p_end[i])
                for i, length in enumerate(lens.tolist())]

        return pad_sequence(sequences, batch_first=True)

    def _e_step(self, emit_probs, trans_probs, mask):
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
        batch_size, seq_len, n_cpos = emit_probs.shape
        alpha, logP_f = self._forward(emit_probs, trans_probs, mask)
        beta,  logP_b = self._forward(emit_probs, trans_probs, mask, forward=False)

        # gamma: [batch_size, seq_len, n_cpos]. aka. posterior probability.
        gamma = (alpha + beta).softmax(-1)
        gamma.masked_fill_(torch.isnan(gamma), 0)

        if seq_len > 1:
            # xi:    [batch_size, seq_len-1, n_cpos_t, n_cpos_t+1]
            xi = alpha[:, :-1].unsqueeze(-1) + \
                    trans_probs.reshape(1, 1, n_cpos, n_cpos) + \
                    emit_probs[:, 1:].unsqueeze(-2) + \
                    beta[:, 1:].unsqueeze(-2)

            xi = xi.contiguous().view(batch_size, seq_len-1, -1)
            xi = xi.softmax(-1).view(batch_size, seq_len-1, n_cpos, n_cpos)

            xi.masked_fill_(torch.isnan(xi), 0)
        else:
            xi = None

        return gamma, xi, (logP_b+logP_f)/2

    def _m_step(self, words, mask, gamma, xi):
        # gamma:         [batch_size, seq_len, n_cpos]
        # xi:            [batch_size, seq_len-1, n_cpos_t, n_cpos_t+1]
        n_words = self.args.n_words

        # one_hot_words: [batch_size, seq_len, n_words]
        one_hot_words = one_hot(words, num_classes=n_words).double()
        one_hot_words.masked_fill_(~mask.unsqueeze(-1), 0.)

        self.gamma_sum  += gamma.sum((0, 1))
        self.start_sum  += gamma[:, 0].sum(0)
        self.end_sum    += gamma[:, -1].sum(0)
        self.emit_sum   += torch.einsum("bsw,bsp->bwp", one_hot_words, gamma).sum(0)
        self.xi_sum     += xi.sum((0, 1)) if xi is not None else 0.
        self.total_sent += words.shape[0]
        return

    def zero_cache(self):
        self.gamma_sum  = 0
        self.start_sum  = 0
        self.end_sum    = 0
        self.emit_sum   = 0
        self.xi_sum     = 0
        self.total_sent = 0

    def step(self):
        self.E.data     = (self.emit_sum / self.gamma_sum).log().float()
        self.start.data = (self.start_sum / self.total_sent).log().float()
        self.end.data   = (self.end_sum / self.total_sent).log().float()
        self.T.data     = (self.xi_sum / self.gamma_sum).log().float()
        return


class VAEPOSModel(nn.Module):
    def __init__(self, 
                 n_words,
                 n_feats,
                 n_cpos,
                 n_tgt_words,
                 n_tgt_nums,
                 n_tgt_hyps,
                 n_tgt_caps,
                 n_tgt_usufs,
                 n_tgt_bsufs,
                 n_tgt_tsufs,
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
                 n_mlp_dec=100,
                 mlp_dropout=.33,
                 dec_dropout=.33,
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
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm_f = nn.LSTM(input_size=n_embed+n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_b = nn.LSTM(input_size=n_embed+n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.layer_norm_0 = nn.LayerNorm(n_lstm_hidden*2, eps=1e-12)
        self.fc_pos = nn.Linear(n_lstm_hidden*2, n_cpos)
        self.layer_norm_1 = nn.LayerNorm(n_cpos, eps=1e-12)

        self.tgt_nums_gen  = nn.Parameter(torch.zeros(n_tgt_nums,  n_mlp_dec))
        self.tgt_hyps_gen  = nn.Parameter(torch.zeros(n_tgt_hyps,  n_mlp_dec))
        self.tgt_caps_gen  = nn.Parameter(torch.zeros(n_tgt_caps,  n_mlp_dec))
        self.tgt_usufs_gen = nn.Parameter(torch.zeros(n_tgt_usufs, n_mlp_dec))
        self.tgt_bsufs_gen = nn.Parameter(torch.zeros(n_tgt_bsufs, n_mlp_dec))
        self.tgt_tsufs_gen = nn.Parameter(torch.zeros(n_tgt_tsufs, n_mlp_dec))
        self.batch_norm = nn.BatchNorm1d(n_mlp_dec*6+n_embed+n_feat_embed, eps=1e-12)
        self.fc_dec = nn.Linear(n_mlp_dec*6+n_embed+n_feat_embed, n_cpos)
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.s_emit = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.tgt_nums_gen.data,  0, 1)
        nn.init.normal_(self.tgt_hyps_gen.data,  0, 1)
        nn.init.normal_(self.tgt_caps_gen.data,  0, 1)
        nn.init.normal_(self.tgt_usufs_gen.data, 0, 1)
        nn.init.normal_(self.tgt_bsufs_gen.data, 0, 1)
        nn.init.normal_(self.tgt_tsufs_gen.data, 0, 1)
        # for w in self.lstm_f.parameters():
        #     nn.init.uniform_(w, 0, 1./(2.*self.args.n_lstm_hidden))
        # for w in self.lstm_b.parameters():
        #     nn.init.uniform_(w, 0, 1./(2.*self.args.n_lstm_hidden))
        # for k in range(self.args.n_lstm_layers):
        #     nn.init.zeros_(self.lstm_f.__getattr__(f'bias_hh_l{k}'))
        #     nn.init.zeros_(self.lstm_b.__getattr__(f'bias_hh_l{k}'))
        #     nn.init.zeros_(self.lstm_f.__getattr__(f'bias_ih_l{k}'))
        #     nn.init.zeros_(self.lstm_b.__getattr__(f'bias_ih_l{k}'))
        #     nn.init.ones_(self.lstm_f.__getattr__(f'bias_hh_l{k}')[1])
        #     nn.init.ones_(self.lstm_b.__getattr__(f'bias_hh_l{k}')[1])
        #     nn.init.ones_(self.lstm_f.__getattr__(f'bias_ih_l{k}')[1])
        #     nn.init.ones_(self.lstm_b.__getattr__(f'bias_ih_l{k}')[1])

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def extra_repr(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        # extra_lines.append('(tgt_words_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_words_gen.shape]) + ')')
        extra_lines.append('(tgt_nums_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_nums_gen.shape]) + ')')
        extra_lines.append('(tgt_hyps_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_hyps_gen.shape]) + ')')
        extra_lines.append('(tgt_caps_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_caps_gen.shape]) + ')')
        extra_lines.append('(tgt_usufs_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_usufs_gen.shape]) + ')')
        extra_lines.append('(tgt_bsufs_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_bsufs_gen.shape]) + ')')
        extra_lines.append('(tgt_tsufs_gen): Parameter(' + ', '.join([str(i) for i in self.tgt_tsufs_gen.shape]) + ')')
        return "\n".join(extra_lines)

    def forward(self, words, feats, tgt_words, tgt_word_features, tgt_feats):
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

        x_f = pack_padded_sequence(embed, mask.sum(1), True, False)
        x_f, _ = self.lstm_f(x_f)
        x_f, _ = pad_packed_sequence(x_f, True, total_length=seq_len)
        x_f = self.lstm_dropout(x_f)

        x_b = reverse_padded_sequence(embed, mask.sum(1), True)
        x_b = pack_padded_sequence(x_b, mask.sum(1), True, False)
        x_b, _ = self.lstm_b(x_b)
        x_b, _ = pad_packed_sequence(x_b, True, total_length=seq_len)
        x_b = reverse_padded_sequence(x_b, mask.sum(1), True)
        x_b = self.lstm_dropout(x_b)

        x = torch.cat([x_f[:, :-2], x_b[:, 2:]], dim=-1)
        x = self.layer_norm_0(x)
        # apply MLPs to the BiLSTM output states
        s_tag = self.fc_pos(x)
        s_tag = self.layer_norm_1(s_tag)
        log_tag_probs = s_tag.log_softmax(-1)

        # cal likelihood
        s_emit = self.get_s_emit(tgt_word_features, tgt_feats)

        log_emit_probs = s_emit.log_softmax(0)
        log_emit_probs = nn.functional.embedding(tgt_words, log_emit_probs)

        likelihood = (log_tag_probs + log_emit_probs)
        return likelihood

    def get_s_emit(self, tgt_word_features, tgt_feats):
        if self.s_emit is None or self.training:
            tgt_word_feature = tgt_word_features[0]
            ext_word_features = tgt_word_feature
            # set the indices larger than num_embeddings to unk_index
            if hasattr(self, 'pretrained'):
                ext_mask = tgt_word_feature.ge(self.word_embed.num_embeddings)
                ext_word_features = tgt_word_feature.masked_fill(ext_mask, self.unk_index)

            # get outputs from embedding layers
            word_embed = self.word_embed(ext_word_features).squeeze(1)
            if hasattr(self, 'pretrained'):
                word_embed += self.pretrained(tgt_word_feature).squeeze(1)
            feat_embed = self.feat_embed(tgt_word_features[1]).squeeze(1)
            # concatenate the word and feat representations

            (tgt_nums, tgt_hyps, tgt_caps,
            tgt_usufs, tgt_bsufs, tgt_tsufs) = tgt_feats
            # [batch_size, seq_len, n_cpos]
            s_nums = nn.functional.embedding(tgt_nums, self.tgt_nums_gen)
            s_hyps = nn.functional.embedding(tgt_hyps, self.tgt_hyps_gen)
            s_caps = nn.functional.embedding(tgt_caps, self.tgt_caps_gen)
            s_usufs = nn.functional.embedding(tgt_usufs, self.tgt_usufs_gen)
            s_bsufs = nn.functional.embedding(tgt_bsufs, self.tgt_bsufs_gen)
            s_tsufs = nn.functional.embedding(tgt_tsufs, self.tgt_tsufs_gen)
            emit_embed = torch.cat((word_embed, feat_embed, s_nums, s_hyps, s_caps, s_usufs, s_bsufs, s_tsufs), dim=-1)
            emit_embed = self.batch_norm(emit_embed)
            self.s_emit = self.fc_dec(emit_embed)
        return self.s_emit



    def decode(self, likelihood):
        return likelihood.argmax(-1)

    def loss(self, likelihood, mask):
        """
        Args:
            s_tag (torch.Tensor): [batch_size, seq_len, n_cpos]
                The scores of all possible arcs.
            words (torch.Tensor): [batch_size, seq_len]
                The scores of all possible labels on each arc.
            mask (torch.BoolTensor): [batch_size, seq_len]
                Mask for covering the unpadded tokens.
        Returns:
            loss (torch.Tensor): scalar
                The training loss.
        """
        expect = -likelihood.logsumexp(-1)
        expect = expect.masked_fill(~mask, 0).sum(-1)
        return expect