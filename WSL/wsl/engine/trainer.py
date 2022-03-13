# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import math
import os
import time
from collections import OrderedDict

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from fvcore.nn.precise_bn import get_bn_modules
from wsl.data import build_detection_test_loader, build_detection_train_loader


class DefaultTrainer_WSL(DefaultTrainer):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        cfg = DefaultTrainer_WSL.auto_scale_workers(cfg, comm.get_world_size())
        super().__init__(cfg)

        self.iter_size = cfg.WSL.ITER_SIZE

        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS

        self.amp = cfg.SOLVER.AMP.ENABLED

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=2
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP50"))
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def run_step(self):
        if self.amp:
            return self.run_step_amp()

        self._trainer.iter = self.iter
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._trainer._data_loader_iter)

            if not self.filter_empty or all([len(x["instances"]) > 0 for x in data]):
                break

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        if self.iter_size > 1:
            losses = losses / self.iter_size
        losses.backward()

        self._trainer._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.iter % self.iter_size == 0:
            self.optimizer.step()

            self.optimizer.zero_grad()

        # del losses
        # del loss_dict
        # torch.cuda.empty_cache()

    def run_step_amp(self):
        self._trainer.iter = self.iter
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()

        while True:
            data = next(self._trainer._data_loader_iter)

            if not self.filter_empty or all([len(x["instances"]) > 0 for x in data]):
                break

        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        if self.iter_size > 1:
            losses = losses / self.iter_size
        self._trainer.grad_scaler.scale(losses).backward()

        self._trainer._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.iter % self.iter_size == 0:
            self._trainer.grad_scaler.step(self.optimizer)
            self._trainer.grad_scaler.update()

            self.optimizer.zero_grad()

        # del losses
        # del loss_dict
        # torch.cuda.empty_cache()

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # if cfg.WSL.SP_ON:
        #     return build_detection_train_loader_sp(cfg)
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        # if cfg.WSL.SP_ON:
        #     return build_detection_test_loader_sp(cfg, dataset_name)
        return build_detection_test_loader(cfg, dataset_name)

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        if old_world_size < num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / scale
        iter_size = cfg.WSL.ITER_SIZE = math.ceil(cfg.WSL.ITER_SIZE / scale)
        logger = logging.getLogger("wsl")
        logger.info(f"Auto-scaling the config to iter_size={iter_size}, learning_rate={lr}.")

        if frozen:
            cfg.freeze()
        return cfg
