# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Source: https://github.com/microsoft/human-pose-estimation.pytorch/
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Adapted by Helena Russello (helena@russello.dev)
# ------------------------------------------------------------------------------

import numpy as np
from utils.data_utils import get_keypoints, get_keypoints_batch
import torch

def calc_dists(preds, target, normalize):
    dists = torch.zeros((preds.shape[0], preds.shape[1]))
    for b in range(preds.shape[0]):
        for k in range(preds.shape[1]):
            if target[b, k, 0] > 1 and target[b, k, 1] > 1:
                normed_preds = preds[b, k, :] / normalize[k]
                normed_targets = target[b, k, :] / normalize[k]
                dists[b, k] = torch.norm(normed_preds - normed_targets)
            else:
                dists[b, k] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = torch.ne(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.lt(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def PCK(targets, predictions, thr=0.5, norm=10):
    """
    Calculates the percentage of correct keypoints
    :param targets: the target keypoints
    :param predictions: the predicted HEATMAPS
    :param thr: threshold under which a keypoint is considered correct
    :return: the PCK per batch image, the mean PCK, and the number of correct keypoints per batch image
    """
    h = predictions.shape[2]
    w = predictions.shape[3]
    batch_size = targets.shape[0]
    n_keypoints = targets.shape[1]
    predictions = get_keypoints_batch(predictions)

    normalize = torch.ones((n_keypoints, 2))
    normalize[:, 0] *= h / norm
    normalize[:, 1] *= w / norm

    return _accuracy(targets, predictions, thr, normalize)

def _accuracy(targets, predictions, thr, normalize):
    """
        Calculates the percentage of correct keypoints
        :param targets: the target keypoints
        :param predictions: the predicted HEATMAPS
        :param thr: threshold under which a keypoint is considered correct
        :return: the PCK per batch image, the mean PCK, and the number of correct keypoints per batch image
        """
    dists = calc_dists(predictions, targets, normalize)

    batch_size = targets.shape[0]
    n_keypoints = targets.shape[1]

    acc = torch.zeros((batch_size, n_keypoints))
    cnt = [0] * batch_size

    for b in range(batch_size):
        for k in range(n_keypoints):
            acc_b_k = dist_acc(dists[b, k], thr)
            if acc_b_k > 0:
                acc[b, k] = acc_b_k
                cnt[b] += 1

    return acc, torch.mean(acc), cnt

def PCKh(targets, predictions, thr=0.5, head_index=[12,13]):
    """
    Calculates the PCK-h metric (Percentage of Correct Keypoints with respect to head size).
    :param targets: the target keypoints
    :param predictions: the predicted HEATMAPS
    :param thr: threshold under which a keypoint is considered correct
    :param head_index: index of the head keypoints
    :return: the PCK per batch image, the mean PCK, and the number of correct keypoints per batch image
    """
    batch_size = targets.shape[0]
    n_keypoints = targets.shape[1]
    predictions = get_keypoints_batch(predictions)
    normalize = torch.ones((n_keypoints, 2))

    # for simplicity we only take the size of the first image in the batch
    # Note that for testing, the batch size is always 1, so evaluation results remain correct
    head_size = torch.norm(targets[0, head_index[0]] - targets[0, head_index[1]]) # euclidian norm
    thr = head_size * thr

    return _accuracy(targets, predictions, thr, normalize)

def euclidian_distance_error(targets, predictions, mean=True):
    """Euclidian distance between the ground truth and prediction for a batch
    targets are the ground truth keypoints ( not heatmaps)
    predictions are the predicted heatmaps
    """
    predictions = get_keypoints_batch(predictions)
    eucl_dist = torch.zeros((targets.size(0), targets.size(1)))
    for b in range(targets.size(0)):
        for j in range(targets.size(1)):
            if targets[b, j, 0] < 0 or targets[b, j, 1] < 0:
                continue
            eucl_dist[b, j] = (torch.sqrt(torch.sum((targets[b, j] - predictions[b, j]) ** 2)))

    # eucl_dist = torch.sqrt((targets - predictions)**2)
    if mean:
        return torch.mean(eucl_dist)
    return eucl_dist