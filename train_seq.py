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

import os
from datetime import datetime

# scipy imports
import numpy as np
import matplotlib.pyplot as plt

# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.seq_pose_dataset import SequentialPoseDataset
from torch.optim import Adam, lr_scheduler, SGD

# Package imports
import core.config
from models.tleap import TLEAP
from utils.plotting_utils import show_heatmaps, show_keypoints
from utils.data_utils import get_keypoints, get_keypoints_batch, dataset_split
from utils.train_utils import save_model, load_model, seed_all, seed_worker
from core.evaluate import euclidian_distance_error, PCKh


def validate(model, criterion, data_loader, config, show=False, save=False, PCK=False, save_path="."):
    """
    Evaluate the model on unseen data
    :param model: the model to evaluate
    :param criterion: the loss function
    :param data_loader: validation or test loader
    :param config: configuration file
    :param show: show plots
    :param save: save plots
    :return: the accuracy, loss and plots
    """
    losses = []
    test_accuracies = []
    test_PCKh = {k: [] for k in range(11)}
    figures = []
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):

            inputs = data['seq'].to(config.device, dtype=torch.float)
            targets = data['heatmaps'][:,-1].to(config.device, dtype=torch.float)
            keypoints = data['keypoints'][:,-1]

            test_outputs = model(inputs)
            test_accuracies.append(euclidian_distance_error(keypoints, test_outputs))

            if PCK:
                for thr in test_PCKh.keys():
                    ckh, pckh, nckh = PCKh(keypoints, test_outputs, thr=thr / 10)  # dict keys are ints
                    test_PCKh[thr].append(pckh)

            losses.append(criterion(test_outputs, targets))

            if show or save:
                keypoints_pred = get_keypoints(test_outputs[0])
                figure = show_keypoints(data['seq'][0][-1], keypoints_pred.cpu(),
                                        save=save, save_fname=os.path.join(save_path, 'test_'+str(i)+'.png'), cmap='gray', tb=(not show))
                if config.wandb:
                    figures.append(
                        wandb.Image(figure))

    if PCK:
        for thr in test_PCKh.keys():
            test_PCKh[thr] = torch.mean(torch.tensor(test_PCKh[thr]))

    return torch.mean(torch.tensor(test_accuracies)), torch.mean(torch.tensor(losses)), figures, test_PCKh

def main():
    """
    Main function, where the magic happens.
    :return: None.
    """
    config, _ = core.config.parse_args("Train Sequential Cowpose")

    # Set the seeds
    seed_all(config.seed)

    # Tensorboard summaries
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    #WandB (Weights and Biases) init
    if config.wandb:
        import wandb
        run = wandb.init(project="cowpose", group=config.group)

        # WandB – Config is a variable that holds and saves hyperparameters and inputs
        wconfig = wandb.config  # Initialize config
        wconfig.batch_size = config.batch_size  # input batch size for training (default: 64)
        wconfig.test_batch_size = 1  # input batch size for testing (default: 1000)
        wconfig.epochs = config.epochs  # number of epochs to train (default: 10)
        wconfig.lr = config.lr  # learning rate (default: 0.01)
        wconfig.no_cuda = config.device  # disables CUDA training
        wconfig.seed = config.seed  # random seed (default: 42)
        wconfig.log_interval = config.frequent  # how many batches to wait before logging training status
        wconfig.seq_length = config.seq_length
        wconfig.optimizer = config.optimizer
        wconfig.depth = config.depth

        tb_comment = run.id
    else:
        # Appendix to file names for saved models.
        tb_comment = 'LR_%.6f_BATCH_%d_EPOCH_%d_SEQ_%d' % (config.lr, config.batch_size, config.epochs, config.seq_length)
    ###########################
    # DATASET                 #
    ###########################

    # TRAIN SET
    train_transform = [
        SequentialPoseDataset.RandomRotate(10),
        SequentialPoseDataset.BrightnessContrast(brightness=(-100, 100), contrast=(-3, 3)),
    ]
    val_transform = None

    train_dataset = SequentialPoseDataset(video_list=config.dataset_csv,
                                          video_dir=config.data_folder,
                                          labels_dir=os.path.join(config.data_folder, 'labels_csv'),
                                          seq_length=config.seq_length,
                                          n_keypoints=len(config.keypoints),
                                          transform=train_transform,
                                          file_format=config.file_format
                                          )

    test_dataset = SequentialPoseDataset(video_list=config.dataset_test,
                                         video_dir=config.data_folder,
                                         labels_dir=os.path.join(config.data_folder, 'labels_csv'),
                                         seq_length=config.seq_length,
                                         transform=val_transform,
                                         n_keypoints=len(config.keypoints),
                                         file_format=config.file_format
                                         )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, worker_init_fn=seed_worker)

    ###########################
    # MODEL and TRAINING INIT #
    ###########################
    model = TLEAP(in_channels=3, out_channels=len(config.keypoints), seq_length=config.seq_length, depth=config.depth).to(config.device)
    if config.wandb:
        wandb.watch(model, log="all")

    # default optimizer = 'amsgrad'
    if config.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    elif config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr, amsgrad=False, weight_decay=0.01)
    else: # amsgrad
        optimizer = Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=0.01)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=(config.epochs // 10), gamma=0.1)

    criterion = nn.MSELoss(reduction='sum')

    # LOAD MODEL
    if config.load_checkpoint:
        checkpoint = load_model(config.load_checkpoint)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        criterion = checkpoint['loss']

    fig_save_path = os.path.join(config.save_checkpoint, tb_comment + '_' + current_time)
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)

    iterations = 0
    epoch = 0
    epoch_fig_save = False
    train_fig_save_path = ""

    while iterations < config.epochs:

        model.train()
        for step, batch in enumerate(train_loader):
            iterations += 1
            optimizer.zero_grad()

            batch_inputs = batch['seq'].to(config.device, dtype=torch.float)
            batch_targets = batch['heatmaps'].to(config.device, dtype=torch.float)
            batch_keypoints = batch['keypoints']

            # Forward step
            batch_preds = model(batch_inputs)
            loss = criterion(batch_preds, batch_targets[:, -1]) / config.batch_size
            loss.backward()
            optimizer.step()

            if step % config.frequent == 0:
                accuracy = euclidian_distance_error(batch_keypoints[:, -1], batch_preds)
                print("Loss at step %d/%d: %.6f, RMSE: %.2f" % (epoch, step, loss.item(), accuracy))

        accuracy = euclidian_distance_error(batch_keypoints[:, -1], batch_preds)

        # Save val figures every [frequent] epoch
        if config.frequent > 0 and (epoch % config.frequent == 0):
            train_fig_save_path = os.path.join(fig_save_path, 'train_%d' % epoch)
            if not os.path.exists(train_fig_save_path):
                os.mkdir(train_fig_save_path)
            epoch_fig_save = True
        else:
            epoch_fig_save = False

        val_accuracy, val_loss, val_figures, val_PCK = validate(model, criterion, test_loader, config, show=False, PCK=True, save=epoch_fig_save, save_path=train_fig_save_path)
        print("Validation loss at epoch %d: %.6f, RMSE: %.2f, PCKh@0.5: %.2f" % (epoch, val_loss, val_accuracy, val_PCK[5]))

        if config.wandb:
            # Plot progress in wandb
            wandb.log({
                "Examples": val_figures,
                "Train Accuracy": accuracy,
                "Train Loss": loss,
                "Val Accuracy": val_accuracy,
                "Val Loss": val_loss,
                "PCKh@0.5": val_PCK[5]
            })

        plt.close('all')

        if config.save_checkpoint:
            model_saved = save_model(config, model, epoch, optimizer, scheduler, criterion, config.save_checkpoint + tb_comment + '_' + current_time )

        epoch += 1
    # end epochs

    ###########
    # TESTING #
    ###########
    # TESTING
    # Perform the evaluation on the whole test set

    test_RSME, test_loss, test_figures, test_PCK = validate(model, criterion, test_loader, config, show=False, save=True, PCK=True, save_path=fig_save_path)
    print("Test RMSE: %.2f" %(test_RSME))
    print("Test PCKh@[thr]:")
    for thr in test_PCK.keys():
        print("PCKh@%.1f : %.2f" % (thr / 10, test_PCK[thr] * 100, ))

    if config.wandb:
        # Log results into wandb
        wandb.log({
            "Examples": test_figures,
            "Test RMSE": test_RSME,
            "Test Loss": test_loss})

        for thr in test_PCK.keys():
            wandb.log({"PCKh": test_PCK[thr], "thr": thr/10})

    plt.close('all')


if __name__ == '__main__':
    main()
