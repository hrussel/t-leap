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
import matplotlib.pyplot as plt
import torch

def show_heatmaps(image, heatmaps, concat=True, save_fname=None):
    """
    Shows the heatmaps on top of the image with Pyplot.
    :param image: The image to display.
    :param heatmaps: The heatmaps of the image.
    :param concat: Whether to show all the heatmaps in one image, or separately. Default = True.
    :param save_fname: Filename to save the image at. Default = None.
    :return: None.
    """
    if type(image) == torch.Tensor:
        image = image.permute(1,2,0).to('cpu')

    h, w, _ = np.shape(image)

    heatmaps_plt = torch.zeros([h, w])
    heatmaps = heatmaps.to("cpu")
    heatmaps = heatmaps.detach()

    if image.max() > 1:
        image = image.int()

    if concat:
        for i, heatmap in enumerate(heatmaps):
            heatmaps_plt += heatmap

        fig = plt.figure()
        img1 = plt.imshow(image, interpolation='none', cmap='gray')
        img2 = plt.imshow(heatmaps_plt, interpolation='none', cmap='jet', alpha=0.5)

        if save_fname:
            plt.savefig(save_fname)
            plt.close()

    else:
        for i, heatmap in enumerate(heatmaps):
            heatmaps_plt = heatmap

            fig = plt.figure()
            img1 = plt.imshow(image, interpolation='none', cmap='gray')
            img2 = plt.imshow(heatmaps_plt, interpolation='none', cmap='jet', alpha=0.5)

            if save_fname:
                plt.savefig(save_fname+'_%d.png' % i)
                plt.close()


def show_keypoints(image, keypoints, save=False, save_fname='keypoints_plot.png', cmap='viridis', tb=True, colormap=None, figsize=(5,5)):
    """
    Shows the keypoints on top of the image with Pyplot.
    :param image: The image to display.
    :param keypoints: The keypoints of the image
    :param save: Whether to save the image or to simply display it. Default = False.
    :param save_fname: Filename to save the image. Default = 'keypoints_plot.png'
    :param cmap: Deprecated, use the colormap param instead. Cmap of the keypoints. Default = 'viridis'.
    :param tb: For use with Tensorboard. Default = True.
    :param colormap: colormap for the keypoints. If None, default colors are selected (only works with <=17 keypoints. If using more, add your own colormap). Default = None.
    :param figsize: Size of the plt figure.
    :return: The figure.
    """
    if type(image) == torch.Tensor and (image.size()[0] == 3 or image.size()[0] == 1):
        image = image.permute(1, 2, 0).to('cpu')
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 1:
        image = np.reshape(image, np.shape(image)[0:2])
    if type(keypoints) == torch.Tensor:
        keypoints = keypoints.cpu().detach().numpy().astype(np.float)

    fig = plt.figure(figsize=figsize)

    if torch.is_tensor(image):
        image = image.squeeze().detach().numpy()
        if image.max() > 1:
            image = image.astype(np.int)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    if colormap is None:
        colormap = ['navy', 'mediumblue', 'blue',
                    'dodgerblue', 'lightskyblue', 'deepskyblue',
                    'turquoise', 'aquamarine', 'palegreen',
                    'khaki', 'yellow', 'gold',
                    'orange', 'darkorange',
                    'orangered', 'red', 'darkred'
                    ]
        if keypoints.shape[0] != 17:
            colormap="red"

    # Don't display keypoints that are out of bounds (e.g., keypoints at (-1,-1))
    h, w = image.shape[:2]
    # print(type(keypoints))
    # print(keypoints[:, 1] > h)
    keypoints[keypoints[:, 1] > h,] = np.nan
    keypoints[keypoints[:, 1] < 0,] = np.nan
    keypoints[keypoints[:, 0] > w,] = np.nan
    keypoints[keypoints[:, 0] < 0,] = np.nan

    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=30, marker='o', c=colormap)

    plt.tight_layout()
    # plt.savefig("test-img.png", pad_inches=0, transparent=True, bbox_inches='tight')
    if save:
        plt.savefig(save_fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
    elif not tb:
        plt.pause(0.001)

    return fig
