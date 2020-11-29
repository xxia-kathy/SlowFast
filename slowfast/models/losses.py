#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import numpy as np
from typing import Callable, Optional
from torch import Tensor
import torch
import torch.nn.functional as F
# import torch.nn._reduction as _Reduction
# from torch.overrides import has_torch_function, handle_torch_function

class SimLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(SimLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.num_classes = 4

    def get_similarity(self, input, target):
        np_sim = np.identity(self.num_classes)
        # test on 4 label case (live-action and rendered-realistic are similar)
        np_sim[0][1] = 0.5
        np_sim[1][0] = 0.5

        similarity_matrix = torch.tensor(np_sim).cuda()
        return similarity_matrix[torch.argmax(input, 1)][torch.argmax(target)]
    def sim_cross_entropy(self, input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
        # if not torch.jit.is_scripting():
        #     tens_ops = (input, target)
        #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
        #         return handle_torch_function(
        #             cross_entropy, tens_ops, input, target, weight=weight,
        #             size_average=size_average, ignore_index=ignore_index, reduce=reduce,
        #             reduction=reduction)
        # if size_average is not None or reduce is not None:
        #     reduction = _Reduction.legacy_get_string(size_average, reduce)
        return F.nll_loss(torch.log(self.get_similarity(input, target)) + F.log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return self.sim_cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "sim_loss": SimLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
