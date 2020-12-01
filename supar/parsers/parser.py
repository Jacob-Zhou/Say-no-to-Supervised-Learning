# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import supar
import torch
import torch.distributed as dist
from functools import partial
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.logging import init_logger, logger
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
from supar.utils.fn import heatmap
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from collections import Counter
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev,
              buckets=32,
              batch_size=5000,
              lr=2e-3,
              mu=.9,
              nu=.9,
              epsilon=1e-12,
              clip=5.0,
              decay=.75,
              decay_steps=5000,
              weight_decay=1e-6,
              epochs=5000,
              patience=100,
              verbose=True,
              **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Load the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        logger.info(f"\ntrain: {train}\ndev:   {dev}")

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(),
                            args.lr,
                            (args.mu, args.nu),
                            args.epsilon,
                            weight_decay=args.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        logger.info(f"{self.optimizer}\n")

        writer = SummaryWriter(comment=args.path.split("/")[1])


        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        logger.info(f"Init:")
        loss, dev_metric = self._evaluate(dev.loader, writer=writer)
        clusters = dev_metric.clusters
        logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}\n")
        heatmap(clusters.cpu(), list(self.CPOS.vocab.stoi.keys()), f"{args.path}.clusters")

        def closure(epoch):
            return self._evaluate(dev.loader, writer=writer, epoch=epoch)

        def write_params(epoch):
            # writer.add_embedding(self.model.tgt_words_gen, metadata=self.TGT_WORD.vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_nums_gen,  metadata=self.TGT_WORD.feature_fields[0].vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_hyps_gen,  metadata=self.TGT_WORD.feature_fields[1].vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_caps_gen,  metadata=self.TGT_WORD.feature_fields[2].vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_usufs_gen, metadata=self.TGT_WORD.feature_fields[3].vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_bsufs_gen, metadata=self.TGT_WORD.feature_fields[4].vocab.itos, global_step=epoch)
            # writer.add_embedding(self.model.tgt_fsufs_gen, metadata=self.TGT_WORD.feature_fields[5].vocab.itos, global_step=epoch)
            for name, param in self.model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param, epoch)
                writer.add_scalar(name+"/std", param.std(), epoch)
                writer.add_scalar(name+"/mean", param.mean(), epoch)
            writer.flush()

        write_params(0)

        for epoch in range(1, args.epochs + 1):

            start = datetime.now()
            logger.info(f"Epoch {epoch} / {args.epochs}:")

            inner_best_metric = self._train(train.loader, 
                                            closure=partial(closure, epoch=epoch), 
                                            best_metric=best_metric, writer=writer)
            if inner_best_metric > best_metric:
                best_e, best_metric = epoch, inner_best_metric
            write_params(epoch)
            loss, dev_metric = closure(epoch)
            t = datetime.now() - start
            # save the model if it is the best so far
            saved = ""
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    self.save(args.path)
                saved = "(saved)"
            logger.info(f"{'current:':10} - loss: {loss:.4f} - {dev_metric}")
            if self.args.evaluate_step < len(train.loader):
                logger.info(f"{'best:':10} - loss: {loss:.4f} - {best_metric}")
            logger.info(f"{t}s elapsed {saved}\n")
            clusters = dev_metric.clusters
            heatmap(clusters.cpu(), list(self.CPOS.vocab.stoi.keys()), f"{args.path}.clusters")
            elapsed += t
            if best_metric == 1.:
                break
            if epoch - best_e >= args.patience:
                break

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        tag_map = {k:self.CPOS.vocab[v] for k,v in metric.tag_map.items()}
        pprint(tag_map)
        recalled_tags = Counter(tag_map.values())
        unrecalled_tags = set(self.CPOS.vocab.stoi) - set(recalled_tags.keys())
        pprint(recalled_tags)
        pprint(unrecalled_tags)
        gold_tag_map = {self.CPOS.vocab[k]:v for k,v in metric.gold_tag_map.items()}
        pprint(gold_tag_map)
        unrecalled_tag_map = {g:tag_map[gold_tag_map[g]] for g in self.CPOS.vocab.stoi}
        unrecalled_tag_map = {k: v for k, v in unrecalled_tag_map.items() if k != v}
        pprint(unrecalled_tag_map)
        # heatmap(metric.clusters.cpu(), list(self.CPOS.vocab.stoi.keys()), f"{args.path}.evaluate.clusters")
        heatmap(self.model.T.softmax(-1).detach().cpu(), [f"#C{n}#" for n in range(len(self.CPOS.vocab))], f"{args.path}.T.clusters")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Load data fields and model parameters from a pretrained parser.

        Args:
            path (str):
                - a string with the shortcut name of a pre-trained parser defined in supar.PRETRAINED
                  to load from cache or download, e.g., `crf-dep-en`.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loaded parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(path):
            state = torch.load(path)
        else:
            path = supar.PRETRAINED[path] if path in supar.PRETRAINED else path
            state = torch.hub.load_state_dict_from_url(path)
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(normalize_paras=not args.em_alg, **args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': self.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)
