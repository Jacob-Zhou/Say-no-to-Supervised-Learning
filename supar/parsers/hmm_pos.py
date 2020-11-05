# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import POSModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField, FeatureField
from supar.utils.fn import ispunct, heatmap
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ManyToOneAccuracy
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class HMMPOSTagger(Parser):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/
    """

    NAME = 'unsuper-tagging'
    MODEL = POSModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CPOS = self.transform.FORM, self.transform.CPOS

    def train(self, train, dev, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, verbose=True, **kwargs):
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
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=False, verbose=True, **kwargs):
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
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000,
                prob=False, tree=True, proj=False, verbose=True, **kwargs):
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

        bar = progress_bar(loader)
        if self.args.em_alg:
            self.model.zero_cache()
        for words, _ in bar:

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence

            emit_probs, trans_probs = self.model(words, self.WORD.features, mask)

            if self.args.em_alg:
                logP = self.model.baum_welch(words, mask, emit_probs, trans_probs)
            else:
                self.optimizer.zero_grad()
                logP = self.model.get_logP(emit_probs, trans_probs, mask)
                loss = -logP.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
                self.scheduler.step()
            bar.set_postfix_str(f" logP: {logP.mean():.4f}")

        if self.args.em_alg:
            self.model.step()

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_logP, metric = 0, ManyToOneAccuracy(n_clusters=self.args.n_cpos, n_cpos=self.args.n_cpos)

        for words, tags in loader:
            mask = words.ne(self.WORD.pad_index)
            emit_probs, trans_probs = self.model(words, self.WORD.features, mask)
            total_logP += self.model.get_logP(emit_probs, trans_probs, mask).sum()
            tag_preds = self.model.decode(emit_probs, trans_probs, mask).to(tags)
            metric(tag_preds, tags, mask)
        total_logP /= len(loader)

        return total_logP, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        tags = []
        for words, in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            lens = mask.sum(1).tolist()
            emit_probs, trans_probs = self.model(words, self.WORD.features, mask)
            tag_preds = self.model.decode(emit_probs, trans_probs, mask)
            tags.extend(tag_preds[mask].split(lens))

        tags = [[f"#C{t}#" for t in seq.tolist()] for seq in tags]
        preds = {'tags': tags}

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        """
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The created parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Build the fields")
        WORD = FeatureField('words', pad="0,0,0,0,0,0,0", unk="1,0,0,0,1,1,1", lower=True)
        CPOS = Field('tags')
        transform = CoNLL(FORM=WORD, CPOS=CPOS)

        train = Dataset(transform, args.train)
        # WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None), not_extend_vocab=True)
        WORD.build(train, args.min_freq)
        CPOS.build(train)
        args.update({
            'n_features': WORD.vocab.n_init,
            'n_words':    WORD.word_vocab.n_init,
            'n_suffix_unigrams': WORD.suffix_unigram_vocab.n_init,
            'n_suffix_bigrams':  WORD.suffix_bigram_vocab.n_init,
            'n_suffix_trigrams': WORD.suffix_trigram_vocab.n_init,
            'n_cpos': CPOS.vocab.n_init,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
        })
        model = cls.MODEL(normalize_paras=not args.em_alg, **args)
        if args.em_alg:
            model.requires_grad_(False)
        # model.load_pretrained(WORD.embed).to(args.device)
        model.to(args.device)
        return cls(args, model, transform)
