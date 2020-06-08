# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

from .scalar_mix import ScalarMix


class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, pad_index=0,
                 requires_grad=False):
        super(BertEmbedding, self).__init__()

        self.bert = BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.pad_index = pad_index
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size

        self.scalar_mix = ScalarMix(n_layers)
        if self.hidden_size != n_out:
            self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords):
        batch_size, seq_len, fix_len = subwords.shape
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))

        if not self.requires_grad:
            self.bert.eval()
        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), True)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), True)
        # return the hidden states of all layers
        _, _, bert = self.bert(subwords, attention_mask=bert_mask)
        # [n_layers, batch_size, n_subwords, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, n_subwords, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size)
        embed = embed.masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed