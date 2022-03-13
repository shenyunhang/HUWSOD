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
from detectron2.utils.events import get_event_storage
from fvcore.nn.precise_bn import get_bn_modules
from wsl.data import build_detection_test_loader, build_detection_train_loader
from wsl.engine.hooks import ParametersNormInspectHook
from wsl.solver import build_optimizer_with_prefix_suffix


class DefaultTrainer_WSL_HUWSOD(DefaultTrainer):
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
        cfg = DefaultTrainer_WSL_HUWSOD.auto_scale_workers(cfg, comm.get_world_size())
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)

        self.iter_size = cfg.WSL.ITER_SIZE

        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS

        self.vis_period = cfg.VIS_PERIOD
        self.data_debug = cfg.WSL.DATA.DEBUG
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(os.path.join(self.output_dir, "vis"), exist_ok=True)

        self.amp = cfg.SOLVER.AMP.ENABLED

        self.K = cfg.WSL.OICR.REFINE_NUM
        self.ema_paramters_keys = []
        self.ema_wsddn_cls_keys = []
        self.ema_decay = cfg.WSL.EMA.DECAY
        self.decay_fun = None
        if cfg.WSL.EMA.DYNAMIC:
            self.decay_fun = lambda x: self.ema_decay * (
                1 - math.exp(-x / 2000)
            )  # decay exponential ramp (to help early epochs)
        for name, param in self.model.named_parameters():
            if "box_refinery" in name:
                self.ema_paramters_keys.append(name)
            if "box_predictor.cls" in name:
                self.ema_wsddn_cls_keys.append(name)

        if comm.is_main_process():
            self.logger.info(f"EMA update key: {self.ema_paramters_keys}")
            self.logger.info(f"EMA update key: {self.ema_wsddn_cls_keys}")

        self.store_scheduler_optimizer()

    @torch.no_grad()
    def ema_step_teacher2student(self, decay):
        # print('before-------------------')
        # print(self.model.roi_heads.box_refinery_1.cls_score.weight)
        self.storage.put_scalar("ema", decay)
        state_dict = self.model.state_dict()
        for i in range(0, self.K - 1):
            for j in range(0, 4):
                teacher_key = self.ema_paramters_keys[i * 4 + j]
                student_key = self.ema_paramters_keys[(i + 1) * 4 + j]
                state_dict[student_key] = (
                    state_dict[teacher_key] * (1 - decay) + state_dict[student_key] * decay
                )
                if self.iter % 1280 == 0 and comm.is_main_process():
                    self.logger.info(f"EMA {decay} update: {teacher_key} --> {student_key}")
        self.model.load_state_dict(state_dict)
        # print('after-------------------')
        # print(self.model.roi_heads.box_refinery_1.cls_score.weight)

    @torch.no_grad()
    def ema_step_student2teacher(self, decay):
        # print('before-------------------')
        # print(self.model.roi_heads.box_refinery_1.cls_score.weight)
        self.storage.put_scalar("ema", decay)
        state_dict = self.model.state_dict()
        for i in range(self.K - 1, 0, -1):
            for j in range(0, 4):
                student_key = self.ema_paramters_keys[i * 4 + j]
                teacher_key = self.ema_paramters_keys[(i - 1) * 4 + j]
                state_dict[teacher_key] = (
                    state_dict[student_key] * (1 - decay) + state_dict[teacher_key] * decay
                )
                if self.iter % 1280 == 0 and comm.is_main_process():
                    self.logger.info(f"EMA {decay} update: {student_key} --> {teacher_key}")
        self.model.load_state_dict(state_dict)
        # print('after-------------------')
        # print(self.model.roi_heads.box_refinery_1.cls_score.weight)

    @torch.no_grad()
    def ema_step_student2teacher_2wsddn_cls(self, decay):
        self.storage.put_scalar("ema", decay)
        state_dict = self.model.state_dict()
        for i in range(self.K - 1, 0, -1):
            for j in range(0, 4):
                student_key = self.ema_paramters_keys[i * 4 + j]
                teacher_key = self.ema_paramters_keys[(i - 1) * 4 + j]
                state_dict[teacher_key] = (
                    state_dict[student_key] * (1 - decay) + state_dict[teacher_key] * decay
                )
                if self.iter % 1280 == 0 and comm.is_main_process():
                    self.logger.info(f"EMA {decay} update: {student_key} --> {teacher_key}")
        for j in range(0, 2):
            student_key = self.ema_paramters_keys[j]
            teacher_key = self.ema_wsddn_cls_keys[j]
            state_dict[teacher_key] = (
                state_dict[student_key][:-1] * (1 - decay) + state_dict[teacher_key] * decay
            )
            if self.iter % 1280 == 0 and comm.is_main_process():
                self.logger.info(f"EMA {decay} update: {student_key} --> {teacher_key}")
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def ema_step(self, decay):
        # print('before-------------------')
        # print(self.model.roi_heads.box_refinery_1.cls_score.weight)
        self.storage.put_scalar("ema", decay)
        state_dict = self.model.state_dict()

        for j in range(0, 2):
            student_key = self.ema_paramters_keys[j]
            teacher_key = self.ema_wsddn_cls_keys[j]
            state_dict[teacher_key] = (
                state_dict[student_key][:-1] * (1 - decay) + state_dict[teacher_key] * decay
            )

            if self.iter % 1280 == 0 and comm.is_main_process():
                self.logger.info(f"EMA update from {student_key} to {teacher_key}.")

        self.model.load_state_dict(state_dict)

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
            ret.append(ParametersNormInspectHook(100, self.model, p=1))
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def run_step(self):
        self.show_scheduler_optimizer()
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

        if self.iter == 0:
            self.freeze(names=["proposal_generator", "roi_heads"])
        if self.iter == self.cfg.WSL.AE.BURNIN_ITERS:
            self.unfreeze(names=["proposal_generator", "roi_heads"])

        data_strong = []
        data_weak = data
        for d in data:
            d_strong = copy.deepcopy(d)
            d_strong["image"] = d_strong["image_strong_aug"]
            del d["image_strong_aug"]
            del d_strong["image_strong_aug"]
            data_strong.append(d_strong)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = {}

        if self.iter < self.cfg.WSL.AE.BURNIN_ITERS:
            if self.cfg.WSL.AE.BURNIN_TYPE == "both":
                loss_dict_ae_burning = self.model(data_weak + data_strong)
            elif self.cfg.WSL.AE.BURNIN_TYPE == "weak":
                loss_dict_ae_burning = self.model(data_weak)
            elif self.cfg.WSL.AE.BURNIN_TYPE == "strong":
                loss_dict_ae_burning = self.model(data_strong)
            else:
                raise NotImplementedError(
                    f"burn in {self.cfg.WSL.AE.BURNIN_TYPE} hasn't implemented"
                )

            for k, v in loss_dict_ae_burning.items():
                if "loss_cls" in k:
                    loss_dict[k + "_" + self.cfg.WSL.AE.BURNIN_TYPE] = v * 1
                else:
                    loss_dict[k + "_" + self.cfg.WSL.AE.BURNIN_TYPE] = v * 1

        else:
            loss_dict_weak, prev_pred_scores_list, prev_pred_boxes_list = self.model(
                data_weak, prev_pred_scores_list=[], prev_pred_boxes_list=[]
            )
            loss_dict_strong, _, _ = self.model(
                data_strong,
                prev_pred_scores_list=prev_pred_scores_list,
                prev_pred_boxes_list=prev_pred_boxes_list,
            )

            for k, v in loss_dict_weak.items():
                if "loss_cls_r" in k:
                    loss_dict[k + "_weak"] = v * self.cfg.WSL.HUWSOD.WEAK_CLS_WEIGHT
                elif "loss_box_reg_r" in k:
                    loss_dict[k + "_weak"] = v * self.cfg.WSL.HUWSOD.WEAK_REG_WEIGHT
                else:
                    loss_dict[k + "_weak"] = v * 1

            for k, v in loss_dict_strong.items():
                if "loss_cls" in k:
                    loss_dict[k + "_strong"] = v * self.cfg.WSL.HUWSOD.STRONG_CLS_WEIGHT
                elif "loss_box_reg" in k:
                    loss_dict[k + "_strong"] = v * self.cfg.WSL.HUWSOD.STRONG_REG_WEIGHT
                else:
                    loss_dict[k + "_strong"] = v * 1

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
            if self.iter >= self.cfg.WSL.AE.BURNIN_ITERS and self.K > 0 and self.cfg.WSL.EMA.ENABLE:
                decay = (
                    self.ema_decay
                    if self.decay_fun is None
                    else self.decay_fun(self.iter - self.cfg.WSL.AE.BURNIN_ITERS + 1)
                )
                if self.cfg.WSL.EMA.DIRECTION == "student2teacher":
                    self.ema_step_student2teacher(decay)
                elif self.cfg.WSL.EMA.DIRECTION == "teacher2student":
                    self.ema_step_teacher2student(decay)
                elif self.cfg.WSL.EMA.DIRECTION == "student2teacher_2wsddn_cls":
                    self.ema_step_student2teacher_2wsddn_cls(decay)
                elif self.cfg.WSL.EMA.DIRECTION == "student2teacher_2wsddn_cls_only":
                    self.ema_step(decay)
                else:
                    raise NotImplementedError(
                        f"ema {self.cfg.WSL.EMA.DIRECTION} hasn't implemented"
                    )
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

        if self.data_debug:
            import cv2

            while True:
                break
                data = next(self._trainer._data_loader_iter)
                for d in data:
                    print("iter: ", self.iter, d.keys())
                    img = d["image"].cpu().numpy().transpose(1, 2, 0)
                    save_path = os.path.join(
                        self.output_dir,
                        "vis",
                        "i" + str(self.iter) + ".g" + str(comm.get_rank()) + ".png",
                    )
                    cv2.imwrite(save_path, img)

                    img = d["image_strong_aug"].cpu().numpy().transpose(1, 2, 0)
                    save_path = os.path.join(
                        self.output_dir,
                        "vis",
                        "i" + str(self.iter) + ".g" + str(comm.get_rank()) + ".strong.png",
                    )
                    cv2.imwrite(save_path, img)

                    self.iter += 1

            data_strong = []
            for d in data:
                d_strong = copy.deepcopy(d)
                d_strong["image"] = d_strong["image_strong_aug"]
                del d["image_strong_aug"]
                del d_strong["image_strong_aug"]
                data_strong.append(d_strong)

            data.extend(data_strong)

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
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer_with_prefix_suffix(cfg, model)

    @torch.no_grad()
    def unfreeze(self, names=[]):
        self.logger.info("Unfreezing {} ...".format(names))

        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group["prefix"] in names:
                self.scheduler.base_lrs[i] = param_group["initial_lr"]
                param_group["lr"] = param_group["initial_lr"]
                param_group["weight_decay"] = param_group["initial_weight_decay"]

                msg = {
                    x: y if x != "params" else [yy.size() for yy in y]
                    for x, y in param_group.items()
                }
                msg = {x: y for x, y in param_group.items() if x != "params"}
                self.logger.info(f"scheduler base_lr {self.scheduler.base_lrs[i]}")
                self.logger.info(f"optimizer param_group {msg}")

    @torch.no_grad()
    def freeze(self, names=[]):
        self.logger.info("Freezing {} ...".format(names))

        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group["prefix"] in names:
                self.scheduler.base_lrs[i] = 0.0
                param_group["lr"] = 0.0
                param_group["weight_decay"] = 0.0

                msg = {
                    x: y if x != "params" else [yy.size() for yy in y]
                    for x, y in param_group.items()
                }
                msg = {x: y for x, y in param_group.items() if x != "params"}
                self.logger.info(f"scheduler base_lr {self.scheduler.base_lrs[i]}")
                self.logger.info(f"optimizer param_group {msg}")

    @torch.no_grad()
    def store_scheduler_optimizer(self):
        self.logger.info("Storing optimizer initial")

        for i, param_group in enumerate(self.optimizer.param_groups):
            assert "initial_lr" in param_group
            assert param_group["initial_lr"] == self.scheduler.base_lrs[i]

            assert "initial_weight_decay" not in param_group
            param_group["initial_weight_decay"] = param_group["weight_decay"]

            msg = {
                x: y if x != "params" else [yy.size() for yy in y] for x, y in param_group.items()
            }
            msg = {x: y for x, y in param_group.items() if x != "params"}
            self.logger.info(f"optimizer param_group {msg}")

    @torch.no_grad()
    def show_scheduler_optimizer(self):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        self.logger.info("scheduler and optimizer")

        for i, param_group in enumerate(self.optimizer.param_groups):
            self.logger.info(f"scheduler base_lr {self.scheduler.base_lrs[i]}")
            msg = {
                x: y if x != "params" else [yy.size() for yy in y] for x, y in param_group.items()
            }
            msg = {x: y for x, y in param_group.items() if x != "params"}
            self.logger.info(f"optimizer param_group {msg}")

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
        logger = logging.getLogger(__name__)
        logger.info(f"Auto-scaling the config to iter_size={iter_size}, learning_rate={lr}.")

        if frozen:
            cfg.freeze()
        return cfg
