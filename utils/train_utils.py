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

import argparse
import core.config
import torch
import os
import numpy
import random

def seed_all(seed):
    """
    Sets the random seed everywhere. See: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    :param seed: The random seed. Default = 10.
    :return: None.
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def save_model(config, model, epoch, optimizer, scheduler, loss, path):
    """
    Save the model weights and the status of training to a pickle file
    :param model: the model to save
    :param epoch: the epoch we were at
    :param optimizer: the optimizer
    :param loss: the loss function
    :param path: the path where to save
    :return: The path where the state was saved
    """
    torch.save({
        'config': config,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss
    }, path)
    return path


def load_model(path):
    """
    Loads a checkpoint of a trained model
    :param path: the path of the pickle file
    :return: epoch, model optimizer and criterion
    """
    return torch.load(path)
