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

    def __init__(self, args, model, transform, optimizer=None, scheduler=None):
        self.args = args
        self.model = model
        self.transform = transform
        if optimizer and scheduler:
            self.optimizer = optimizer
            self.scheduler = scheduler
        else:
            self.optimizer, self.scheduler = self.build_optim(model, **args)

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
        if not self.optimizer or not self.scheduler:
            self.optimizer, self.scheduler = self.build_optim(self.model, **args)
        logger.info(f"{self.optimizer}\n")

        elapsed = timedelta()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_likelihood = float('inf')
        best_restart = 0
        for n in range(0, args.restarts):
            self.model = self.MODEL(**args)
            self.model.load_pretrained(self.WORD.embed).to(device)
            self.optimizer, self.scheduler = self.build_optim(self.model, **args)
            if dist.is_initialized():
                self.model = DDP(self.model,
                                device_ids=[dist.get_rank()],
                                find_unused_parameters=True)
            for epoch in range(1, args.restart_epochs + 1):
                start = datetime.now()
                logger.info(f"Restart {n+1 :<4d} / {args.restarts}:")
                logger.info(f"Epoch   {epoch :<4d} / {args.restart_epochs}:")
                self._train(train.loader, epoch=epoch)
                likelihood, _ = self._evaluate(dev.loader)
                t = datetime.now() - start
                # save the model if it is the best so far
                saved = ""
                if likelihood < best_likelihood:
                    best_likelihood, best_restart = likelihood, n
                    if is_master():
                        self.save(args.path+"_init")
                    saved = "(saved)"
                logger.info(f"{'current:':10} - likelihood: {-likelihood:.4f}")
                logger.info(f"{t}s elapsed {saved}\n")

        writer = SummaryWriter(comment=args.path.split("/")[1])

        logger.info(f"Load best initialization: {best_restart}\n")
        if args.restarts > 0:
            best_parser = self.load(args.path+"_init")
            self.model = best_parser.model
            self.optimizer = best_parser.optimizer
            self.scheduler = best_parser.scheduler
        loss, dev_metric = self._evaluate(dev.loader, writer=writer)
        clusters = dev_metric.clusters
        logger.info(f"{'dev:':6} - likelihood: {loss:.4f} - {dev_metric}\n")
        heatmap(clusters.cpu(), list(self.CPOS.vocab.stoi.keys()), f"{args.path}.clusters")

        best_e, best_metric = args.restart_epochs, dev_metric

        def closure(epoch):
            return self._evaluate(dev.loader, writer=writer, epoch=epoch)

        def write_params(epoch):
            for name, param in self.model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param, epoch)
                writer.add_scalar(name+"/std", param.std(), epoch)
                writer.add_scalar(name+"/mean", param.mean(), epoch)
            writer.flush()

        write_params(0)

        for epoch in range(args.restart_epochs + 1, args.epochs + 1):

            start = datetime.now()
            logger.info(f"Epoch   {epoch :<4d} / {args.epochs}:")

            inner_best_metric = self._train(train.loader, 
                                            closure=partial(closure, epoch=epoch), 
                                            best_metric=best_metric, epoch=epoch)
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
        heatmap(metric.clusters.cpu(), list(self.CPOS.vocab.stoi.keys()), f"{args.path}.evaluate.clusters")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
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
    def build_optim(cls, 
                    model,
                    lr=2e-3,
                    mu=.9,
                    nu=.9,
                    epsilon=1e-12,
                    clip=5.0,
                    decay=.75,
                    decay_steps=5000,
                    weight_decay=1e-6,
                    **kwargs):
        optimizer = Adam(model.parameters(),
                              lr,
                              (mu, nu),
                              epsilon,
                              weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, decay**(1/decay_steps))
        return optimizer, scheduler

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
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        optimizer, scheduler = cls.build_optim(model, **args)
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        transform = state['transform']
        return cls(args, model, transform,
                   optimizer=optimizer,
                   scheduler=scheduler)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        if 'optimizer' in self.__dict__:
            optimizer = self.optimizer
            optimizer_state_dict = optimizer.state_dict()
        if 'scheduler' in self.__dict__:
            scheduler = self.scheduler
            scheduler_state_dict = scheduler.state_dict()
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': self.args,
                 'state_dict': state_dict,
                 'optimizer_state_dict': optimizer_state_dict,
                 'scheduler_state_dict': scheduler_state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)
