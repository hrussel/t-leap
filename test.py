import os
from datetime import datetime

# scipy imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def test(model, criterion, data_loader, config, show=False, save=False, PCK=False, save_path="."):
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
    test_rmse = []
    test_PCKh = {k: [] for k in range(11)}
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):

            inputs = data['seq'].to(config.device, dtype=torch.float)
            targets = data['heatmaps'][:,-1].to(config.device, dtype=torch.float)
            keypoints = data['keypoints'][:,-1]

            test_outputs = model(inputs)
            test_rmse.append(euclidian_distance_error(keypoints, test_outputs).item())

            if PCK:
                for thr in test_PCKh.keys():
                    ckh, pckh, nckh = PCKh(keypoints, test_outputs, thr=thr / 10)  # dict keys are ints
                    test_PCKh[thr].append(pckh.item())

            losses.append(criterion(test_outputs, targets).item())

            if show or save:
                keypoints_pred = get_keypoints(test_outputs[0])
                _ = show_keypoints(data['seq'][0][-1], keypoints_pred.cpu(),
                                        save=save, save_fname=os.path.join(save_path, 'test_'+str(i)+'.png'), cmap='gray', tb=(not show))

    return test_rmse, losses, test_PCKh


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


    # Appendix to file names for saved models.
    tb_comment = 'LR_%.6f_BATCH_%d_EPOCH_%d_SEQ_%d' % (config.lr, config.batch_size, config.epochs, config.seq_length)
    ###########################
    # DATASET                 #
    ###########################

    val_transform = None

    test_dataset = SequentialPoseDataset(video_list=config.dataset_test,
                                         video_dir=config.data_folder,
                                         labels_dir=os.path.join(config.data_folder, 'labels_csv'),
                                         seq_length=config.seq_length,
                                         transform=val_transform,
                                         n_keypoints=len(config.keypoints),
                                         file_format=config.file_format
                                         )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, worker_init_fn=seed_worker)

    ###########################
    # MODEL and TRAINING INIT #
    ###########################
    model = TLEAP(in_channels=3, out_channels=len(config.keypoints), seq_length=config.seq_length, depth=config.depth).to(config.device)


    optimizer = Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=0.01)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=(config.epochs // 10), gamma=0.1)

    criterion = nn.MSELoss(reduction='sum')

    # LOAD MODEL
    checkpoint_path = config.load_checkpoint
    if config.load_checkpoint:
        checkpoint = load_model(config.load_checkpoint)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        criterion = checkpoint['loss']
    else:
        print("Please specify a checkpoint for testing.")
        exit(1)

    checkpoint_path = checkpoint_path.split('.model')[0]
    fig_save_path = os.path.join(checkpoint_path, 'test')
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)


    ###########
    # TESTING #
    ###########
    # TESTING
    # Perform the evaluation on the whole test set
    print("Saving test images to %s" % fig_save_path)

    test_RSME, test_loss, test_PCK = test(model, criterion, test_loader, config, show=False, save=True, PCK=True, save_path=fig_save_path)
    results_dict = {"Test_ID": [*range(len(test_dataset)), 'mean']}
    results_dict["Test_RMSE"] =  [*test_RSME, np.mean(test_RSME)]

    for thr in test_PCK.keys():
        results_dict[f"Test_PCKh@{thr/10}"] = [*test_PCK[thr], np.mean(test_PCK[thr])]

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(os.path.join(fig_save_path, 'test_metrics.csv'), index=False)

    plt.close('all')


if __name__ == '__main__':
    main()
