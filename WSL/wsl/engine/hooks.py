# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from detectron2.engine.train_loop import HookBase

__all__ = [
    "ParametersNormInspectHook",
]


"""
Implement some common hooks.
"""


class ParametersNormInspectHook(HookBase):
    def __init__(self, period, model, p):
        self._period = period
        self._model = model
        self._p = p

    @torch.no_grad()
    def _do_inspect(self):
        for key, val in self._model.named_parameters(recurse=True):
            self.trainer.storage.put_scalar(
                "parameters norm {}/{}".format(self._p, key), torch.norm(val, p=self._p)
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter != self.trainer.max_iter:
                self._do_inspect()
