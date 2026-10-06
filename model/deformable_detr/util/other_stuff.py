

'''
Set fo fuctions copied from Deformable Detr (https://github.com/fundamentalvision/Deformable-DETR)
'''

import torch.nn.functional as F
import torch
from torch import nn
from model.deformable_detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                                             accuracy, get_world_size, interpolate,
                                             is_dist_avail_and_initialized, inverse_sigmoid)
from model.deformable_detr.util import box_ops
import copy

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    '''
    Copied from Deformable Detr (https://github.com/fundamentalvision/Deformable-DETR)
    Direkt lokation (https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/segmentation.py#L196)
    '''

    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
