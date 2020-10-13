# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.models import VAEDependencyModel
from supar.parsers.biaffine_dependency import BiaffineDependencyParser
from supar.utils import Config
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric

logger = get_logger(__name__)


class VAEDependencyParser(BiaffineDependencyParser):
    """
    The implementation of XXX
    """

    NAME = 'vae-dependency'
    MODEL = VAEDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train, dev, test, buckets=32, batch_size=5000, punct=False,
              mbr=True, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        """
        Args:
            train, dev, test (list[list] or str):
                the train/dev/test data, both list of instances and filename are allowed.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            punct (bool):
                If False, ignores the punctuations during evaluation. Default: False.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            partial (bool):
                True denotes the trees are partially annotated. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False,
                 mbr=True, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        """
        Args:
            data (str):
                The data to be evaluated.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            punct (bool):
                If False, ignores the punctuations during evaluation. Default: False.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            partial (bool):
                True denotes the trees are partially annotated. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False,
                mbr=True, tree=True, proj=True, verbose=True, **kwargs):
        """
        Args:
            data (list[list] or str):
                The data to be predicted, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: None.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            prob (bool):
                If True, outputs the probabilities. Default: False.
            mbr (bool):
                If True, returns marginals for MBR decoding. Default: True.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            A Dataset object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()
        for supervised_mask, words, feats, arcs, rels in bar:
            unsuper_loss = self.args.semi_supervised
            if ~supervised_mask.any() and ~unsuper_loss:
                continue
            self.optimizer.zero_grad()
            batch_size, seq_len = words.shape

            mask = words.ne(self.WORD.pad_index)
            mask[:, 0] = 0
            word_mask  = mask & words.lt(self.args.n_words)
            # ignore the first token of each sentence
            s_arc, s_rel, s_word, kld_loss = self.model(words, feats, supervised_mask, arcs)
            loss = self.model.loss(s_arc, s_rel, s_word, 
                                          arcs, rels, words, 
                                          kld_loss, mask, word_mask,
                                          supervised_mask=supervised_mask)
            # if torch.isnan(loss).any():
                # exit()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for _, words, feats, arcs, rels in loader:
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            batch_size, seq_len = words.shape
            s_arc, s_rel = self.model(words, feats)
            # s_arc, s_rel, s_word, kld_loss = self.model(words, feats, mask.new_zeros(batch_size), arcs)

            loss = torch.zeros(1)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            # arc_preds[~mask] = 0
            # for i in range(20):
            #     # print(tree[i])
            #     print()
            #     print(self.WORD.vocab[words[i][1:]])
            #     print(self.WORD.vocab[s_word.argmax(-1)[i]])
            #     print()
            #     print(arcs[i])
            #     print(arc_preds[i])
            #     print('-------------------')
            # exit()
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            batch_size = words.shape[0]
            s_arc, s_rel = self.model(words, feats)
            if self.args.mbr:
                s_arc = self.model.crf(s_arc, mask, mbr=True)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc if self.args.mbr else s_arc.softmax(-1)
                probs.extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds
