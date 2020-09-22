# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.models import CRFDependencyModel
from supar.parsers.biaffine_dependency import BiaffineDependencyParser
from supar.utils import Config
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric

logger = get_logger(__name__)


class CRFDependencyParser(BiaffineDependencyParser):
    """
    The implementation of first-order CRF Dependency Parser.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang (ACL'20)
          Efficient Second-Order TreeCRF for Neural Dependency Parsing
          https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    NAME = 'crf-dependency'
    MODEL = CRFDependencyModel

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
            neg_sample_time = self.args.neg_sample
            min_len = (words.ne(self.WORD.pad_index).sum(-1) - 1).min()
            if unsuper_loss and seq_len > 1 and batch_size > 1:
                slice_size = seq_len
                chunk_words, chunk_feats = words.chunk(slice_size, dim=1), feats.chunk(slice_size, dim=1)
                rand_words, rand_feats = [], []
                for cw, cf in zip(chunk_words, chunk_feats):
                    rand_index = torch.randperm(int(batch_size * neg_sample_time)) % batch_size
                    rand_words.append(cw[rand_index])
                    rand_feats.append(cf[rand_index])
                fake_words = torch.cat(rand_words, dim=1)
                fake_feats = torch.cat(rand_feats, dim=1)
                rand_step_index = torch.randperm(min_len)
                fake_words[:, torch.arange(min_len)] = fake_words[:, rand_step_index]
                fake_feats[:, torch.arange(min_len)] = fake_feats[:, rand_step_index]
                words = torch.cat([words, fake_words], dim=0)
                feats = torch.cat([feats, fake_feats], dim=0)
            else:
                unsuper_loss = False

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                          supervised_mask=supervised_mask,
                                          real_batch_size=batch_size,
                                          mbr=self.args.mbr,
                                          partial=self.args.partial,
                                          unsuper_loss=unsuper_loss)
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
            metric(arc_preds[:batch_size], rel_preds[:batch_size], arcs, rels, mask[:batch_size])
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
            
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask,
                                          mbr=self.args.mbr,
                                          partial=self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
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
