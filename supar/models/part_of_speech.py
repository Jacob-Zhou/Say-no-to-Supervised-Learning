# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM,
                           Triaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.modules.treecrf import CRF2oDependency, CRFDependency, MatrixTree
from supar.utils import Config
from supar.utils.alg import eisner, eisner2o, mst, kmeans
from supar.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.functional import one_hot
from supar.utils.fn import heatmap

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
        beta, logP_b  = self._forward(emit_probs, trans_probs, mask, forward=False)

        # gamma: [batch_size, seq_len, n_cpos]
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

