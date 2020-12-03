# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from functools import partial, lru_cache
from supar.models import VAEPOSModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, eos, pad, unk
from supar.utils.field import Field, SubwordField
from supar.utils.fn import ispunct, heatmap, hasnumber_fn, isinitcapitalized_fn, hashyphen_fn, getsuffix_fn, replace_digit_fn
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ManyToOneAccuracy, Metric
from supar.utils.transform import CoNLL
from supar.utils.parallel import is_master

logger = get_logger(__name__)


class VAEPOSTagger(Parser):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/
    """

    NAME = 'unsuper-vae-tagging'
    MODEL = VAEPOSModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        (self.WORD, self.FEAT, self.TGT_WORD), self.CPOS = self.transform.FORM, self.transform.CPOS

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

    @lru_cache
    def get_word_features(self):
        sequences = [[word] for word in self.TGT_WORD.vocab.itos]
        word_features = []
        # batch_size == n_word
        word_features.append(self.WORD.compose([seq[1:-1] for seq in self.WORD.transform(sequences)]))
        word_features.append(self.FEAT.compose([seq[1:-1] for seq in self.FEAT.transform(sequences)]))
        return word_features

    def _train(self, loader, closure=None, best_metric=None, writer=None, epoch=0):
        self.model.train()

        bar = progress_bar(loader)
        step = 0
        evaluate_step = self.args.evaluate_step
        if evaluate_step > len(loader):
            closure = None
        dev_metric = None
        if best_metric is None:
            best_metric = Metric()
        saved = ''
        features = self.TGT_WORD.features
        word_features = self.get_word_features()
        for words, feats, tgt_words, _ in bar:
            self.optimizer.zero_grad()
            mask = words.ne(self.WORD.pad_index)[:, 2:]
            # ignore the first token of each sentence
            likelihood = self.model(words, feats, tgt_words, word_features, features, epoch)
            loss = self.model.loss(likelihood, mask).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            if closure is None:
                bar.set_postfix_str(f" lr: {self.scheduler.get_last_lr()[0]:.4e}, loss: {loss:.4f}")
            else:
                if step != 0 and step % evaluate_step == 0:
                    saved = ''
                    _, dev_metric = closure()
                    self.model.train()
                    if dev_metric > best_metric:
                        best_metric = dev_metric
                        if is_master():
                            self.save(self.args.path)
                            saved = "(saved)"
                bar.set_postfix_str(f" lr: {self.scheduler.get_last_lr()[0]:.4e}, loss: {loss:.4f}, {dev_metric} {saved}")
            step += 1

        return best_metric

    @torch.no_grad()
    def _evaluate(self, loader, writer=None, epoch=0):
        self.model.eval()
        self.model.s_emit = None

        total_loss, metric = 0, ManyToOneAccuracy(n_clusters=self.args.n_cpos, n_cpos=self.args.n_cpos)
        features = self.TGT_WORD.features
        word_features = self.get_word_features()
        for words, feats, tgt_words, tags in loader:
            mask = words.ne(self.WORD.pad_index)[:, 2:]
            likelihood = self.model(words, feats, tgt_words, word_features, features)
            total_loss += self.model.loss(likelihood, mask).sum()
            tag_preds = self.model.decode(likelihood)
            metric(tag_preds, tags, mask)
        total_loss /= len(loader)
        if writer is not None:
            writer.add_scalar('Eval/Loss', total_loss, global_step=epoch)
            writer.add_scalar('Eval/M-1',  metric.score, global_step=epoch)
            writer.flush()
        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        tags = []
        for words, feats, _ in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            lens = mask.sum(1).tolist()
            s_tag = self.model(words, feats)
            tag_preds = self.model.decode(s_tag, words, mask)
            tags.extend(tag_preds[mask].split(lens))

        tags = [[f"#C{t}#" for t in seq.tolist()] for seq in tags]
        preds = {'tags': tags}

        return preds


    @classmethod
    def build(cls, path, min_freq=2, tgt_min_freq=0, fix_len=20, **kwargs):
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
        TGT_NUM = Field('tgt_num', fn=hasnumber_fn)
        TGT_HYP = Field('tgt_hyp', fn=hashyphen_fn)
        TGT_CAP = Field('tgt_cap', fn=isinitcapitalized_fn)
        TGT_USUF = Field('tgt_usuf', fn=partial(getsuffix_fn, n=1, replace_digit=True))
        TGT_BSUF = Field('tgt_bsuf', fn=partial(getsuffix_fn, n=2, replace_digit=True))
        TGT_TSUF = Field('tgt_tsuf', fn=partial(getsuffix_fn, n=3, replace_digit=True))
        WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True, fn=replace_digit_fn)
        # WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=False)
        TGT_WORD = Field('tgt_words', 
                         feature_fields=[TGT_NUM, TGT_HYP, TGT_CAP, 
                                         TGT_USUF, TGT_BSUF, TGT_TSUF])
        CPOS = Field('tags')
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                eos=tokenizer.eos_token or tokenizer.sep_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
           raise NotImplementedError()
        transform = CoNLL(FORM=(WORD, FEAT, TGT_WORD), CPOS=CPOS)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        TGT_WORD.build(train, args.tgt_min_freq)
        FEAT.build(train)
        CPOS.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_tgt_words': TGT_WORD.vocab.n_init,
            'n_tgt_nums':  TGT_NUM.vocab.n_init,
            'n_tgt_hyps':  TGT_HYP.vocab.n_init,
            'n_tgt_caps':  TGT_CAP.vocab.n_init,
            'n_tgt_usufs': TGT_USUF.vocab.n_init,
            'n_tgt_bsufs': TGT_BSUF.vocab.n_init,
            'n_tgt_tsufs': TGT_TSUF.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_cpos':  len(CPOS.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'feat_pad_index': FEAT.pad_index,
            'cpos_pad_index': CPOS.pad_index
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
