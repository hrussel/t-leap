# #############################################################################
# Copyright 2022 Helena Russello
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #############################################################################
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.train_utils import seed_worker

def get_mean_std(dataloader):
    """
    Calculate the mean and stddev of the dataset.
    This assumes that the whole data can't be loaded in RAM.
    Instead of calculating the mean and stddev overall the whole dataset, we average the mean and stddev per batch
    :param dataloader: the pytorch dataloader
    :return: (tuple) mean, std

    source: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples


def get_keypoints_batch(batch_heatmaps, thr=0):
    """
    Gets the keypoints from the heatmaps for the whole batch
    :param batch_keypoints: The heatmaps of the batch
    :param thr: Threshold for the location of the keypoints.
                Default =0. So if the keypoint is located a (0,0), it set as non-existing.
    """
    b, k, w, h = batch_heatmaps.size()
    batch_keypoints = torch.zeros([b, k, 2])

    for i, heatmaps in enumerate(batch_heatmaps):
        keypoints = get_keypoints(heatmaps, thr)
        batch_keypoints[i, :, :] = keypoints
    return batch_keypoints


def get_keypoints(heatmaps, thr=0):
    """
    Gets the keypoints from the heatmaps
    :param heatmaps: The heatmaps of shape (n_keypoints, height, width)
    :param thr: Threshold for the location of the keypoints.
                Default =0. So if the keypoint is located a (0,0), it set as non-existing.
    :return: The keypoints
    """
    n, h, w = heatmaps.size()
    flat = heatmaps.view(n, -1)

    max_val, max_idx = flat.max(dim=1)

    xx = (max_idx % w).view(-1, 1)  # (-1,1) for column vector
    yy = (max_idx // w).view(-1, 1)  # (-1,1) for column vector

    xx[max_val <= thr] = -1
    yy[max_val <= thr] = -1

    keypoints = torch.cat((xx, yy), dim=1)

    return keypoints


def get_heatmaps(sample, sigma=5, normalize=True):
    """
    Generate heatmaps from a sample
    :param sample: The sample from which to generate heatmaps. Dictionary containing an image and keypoints. (Sample returned by the Dataset)
    :param sigma: The standard deviation of the gaussian noise. Default = 5 pixels.
    :param normalize: Whether to normalize the heatmaps
    :return: The heatmaps of the keypoints
    """
    keypoints = sample['keypoints']
    likelihood = np.ones(np.shape(keypoints)[0])
    sample['likelihood'] = likelihood  # Likelihood =1
    return get_heatmaps_likelihood(sample, sigma, normalize)


def get_heatmaps_likelihood(sample, sigma=5, normalize=True):
    """
        Generates heatmaps from the keypoints of a sample. The std-dev depends on the likelihood of each keypoint. If the likelikhood is high, the std-dev will be smaller.
        :param sample: The sample from which to generate heatmaps.
        :param sigma: The standard deviation of the gaussian noise
        :param normalize: Whether to normalize the heatmaps
        :return: The heatmaps of the keypoints
        """
    image, keypoints, likelihood = sample['image'], sample['keypoints'], sample['likelihood']
    h = np.shape(image)[0]
    w = np.shape(image)[1]

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    heatmaps = np.zeros([np.shape(keypoints)[0], h, w])
    keypoints = np.nan_to_num(keypoints, nan=-1)

    for i, keypoint in enumerate(keypoints):
        if keypoint[0] < 0 or keypoint[1] < 0:
            continue
        kx = keypoint[0]
        ky = keypoint[1]
        i_sigma = sigma
        # i_sigma = sigma / likelihood[i]
        # Gaussian distribution with peak at the keypoint annotation
        heatmaps[i] = np.exp(-((yy - ky) ** 2 + (xx - kx) ** 2) / (2 * i_sigma ** 2))

        if not normalize:
            heatmaps[i] /= i_sigma * np.sqrt(2 * np.pi)

    return heatmaps


def get_keypoints_original_size(keypoints, bbox, scaling_factor=1):
    """
    Scales back keypoints from a resized bounding box to the original image
    :param keypoints: the predicted keypoints
    :param bbox: the bounding box on the image
    :param scaling_factor: the scaling factor from the resized bbox to the original bbox
    :return: the coordinates of the keypoints in the original image
    """
    orig_keypoints = torch.full_like(keypoints, -1, dtype=torch.float32).cpu()

    for i, k in enumerate(keypoints):
        if k[0] > 0 and k[1] > 0:
            orig_keypoints[i][0] = k[0].cpu() * scaling_factor + bbox[2]
            orig_keypoints[i][1] = k[1].cpu() * scaling_factor + bbox[0]
        else:
            orig_keypoints[i][0] = 0
            orig_keypoints[i][1] = 0
    return orig_keypoints

def dataset_split(train_dataset, val_dataset, config, k_fold=1):
    """
    Split the train and validation set according to the settings from config.
    If k_fold is specified, divides the dataset into k parts and provide k train and validation loaders
    :param train_dataset: dataset for training
    :param val_dataset: dataset for validation
    :param config: configuration list
    :param k_fold: number of folds. Default: 1
    :return: the train loaders and the validation loaders
    """

    # Shuffle the dataset
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    # Divide the dataset into k equal sets
    len_split = len(train_dataset) // k_fold
    train_split = int(np.floor(config.val_size * len_split))

    train_loaders = []
    val_loaders = []

    # for each fold make a train and validation set
    for split in range(k_fold):
        start_split = split * len_split
        split_indices = indices[start_split:start_split+len_split]

        train_idx, val_idx = split_indices[train_split:], split_indices[: train_split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loaders.append(DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, drop_last=False, worker_init_fn=seed_worker))
        val_loaders.append(DataLoader(val_dataset, batch_size=1, sampler=val_sampler, drop_last=False, worker_init_fn=seed_worker))

    return train_loaders, val_loaders
