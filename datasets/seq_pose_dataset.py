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

import pandas as pd
import numpy as np
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.plotting_utils import show_keypoints
from utils.data_utils import get_heatmaps_likelihood

# noinspection PyShadowingNames
class SequentialPoseDataset(Dataset):

    def __init__(self,
                 video_list,
                 video_dir,
                 labels_dir,
                 seq_length,
                 n_keypoints=17,
                 transform=None,
                 normalize=None,
                 crop_body=True,
                 is_test=False,
                 file_format="%d.png"):
        """
        Constructs a sequential dataset. Each item is a sequence of frames from videos
        :param video_list: csv file listing videos, start and end of labelled frames (video name, start, end)
        :param video_dir: directory where the videos are stored
        :param labels_dir: directory where the csv annotations are stored
        :param seq_length: lenght of the sequences
        :param transform: Optional.
        :param normalize: mean and stdev values for normalization
        """
        self.videos = np.loadtxt(video_list, delimiter=',', dtype=np.unicode, skiprows=1)  # video_name, start, end
        self.video_dir = video_dir
        self.labels_dir = labels_dir
        self.seq_length = seq_length
        self.transform = transform
        self.normalize = normalize
        self.n_keypoints = n_keypoints
        self.crop_body = crop_body

        self.dataset = []
        self.isTest = is_test
        # self.MAX_SEQ = 4
        self.file_format = file_format
        # self.STEP = 4

        if is_test:
            self.crop_widths = {}
        self.videos = self.videos.reshape(-1, 3)
        for video, start, end in self.videos:
            self.dataset.extend([[video, i] for i in range(int(start), int(end) - seq_length + 2)])
            # self.dataset.extend([[video, i] for i in range(int(start), int(end) - self.MAX_SEQ + 2)]) # same number of samples for any seq length
            # self.dataset.extend([[video, i + self.MAX_SEQ - 1] for i in range(int(start), int(end) - self.MAX_SEQ + 2, self.STEP)])  # same number of samples for any seq length

    def __getitem__(self, item):

        video, start = self.dataset[item][0], self.dataset[item][1]
        frames = self._get_video_frames(video, start, self.file_format, opencv=False)
        csvfilenmae = video.split('.')[0]+'.csv'
        labels = pd.read_csv(os.path.join(self.labels_dir,csvfilenmae ))
        seq_labels = labels[(labels['frame'] >= start) & (labels['frame'] < (start + self.seq_length))].iloc[:, 2:]

        # in csv file, keypoints for a part are in format: part1_x, part1_y, part1_likelihood, part2_x, part2_y, ...
        keypoints_idx = [True, True, False] * self.n_keypoints
        seq_keypoints = np.array(seq_labels.iloc[:, keypoints_idx]).astype('float')
        if seq_keypoints.shape[0] == 0:
            print("here")
        seq_keypoints = seq_keypoints.reshape([seq_keypoints.shape[0], -1, 2])  # shape (seq_length, n_keypoints, 2)

        likelihood_idx = [False, False, True] * self.n_keypoints
        seq_likelihood = np.array(seq_labels.iloc[:, likelihood_idx]).astype('float')

        sample = {'seq': frames, 'keypoints': seq_keypoints, 'likelihood': seq_likelihood}

        if self.transform:
            for t in self.transform:
                if isinstance(t, self.CropBody):
                    if self.isTest and (video in self.crop_widths):
                        t.set_crop_width(self.crop_widths[video])
                    else:
                        t.set_crop_width(-1)
                sample = t(sample)
                if isinstance(t, self.CropBody) and self.isTest:
                    self.crop_widths[video] = t.get_crop_width()

        # Generate heatmaps for each frame
        seq_heatmaps = []
        for i, image in enumerate(sample['seq']):
            i_sample = {'image': image, 'keypoints': sample['keypoints'][i], 'likelihood': sample['likelihood'][i]}
            heatmaps = get_heatmaps_likelihood(i_sample)
            seq_heatmaps.append(heatmaps)
        seq_heatmaps = np.stack(seq_heatmaps)
        sample['heatmaps'] = seq_heatmaps

        # Replace missing keypoints by -1 values
        sample['keypoints'] = np.nan_to_num(sample['keypoints'], nan=-1)

        if self.normalize:
            tensor = transforms.Compose(
                [self.ToTensor(), self.Normalize(self.normalize['mean'], self.normalize['std'])])
        else:
            tensor = transforms.Compose([self.ToTensor()])
        sample = tensor(sample)

        sample['video'] = video
        sample['frame'] = start + self.seq_length - 1
        # sample['frame'] = start
        return sample

    def __len__(self):
        return len(self.dataset)

    def _get_video_frames(self, video, start_frame, file_format, opencv=True):
        """
        Retrieves a sequence of frames from a video
        :param video: the video to read the frames from
        :param start_frame: the start frame of the sequence
        :param opencv: whether to extract the frames from the video with opencv
        :return: a list of frames of length seq_length or None if an error occured
        """
        if opencv:
            return self._get_video_frames_opencv(video, start_frame)

        video_name = video.split('.')[0]
        video_path = os.path.join(self.video_dir, video_name)

        frames = []
        # for i in range(self.seq_length):
        #     s = self.seq_length - i
        #     frame_path = os.path.join(video_path, file_format % (start_frame - s + 1))
        for i in range(start_frame, start_frame + self.seq_length):
            frame_path = os.path.join(video_path, file_format % i)
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                print("here FRAME NONE")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames

    def _get_video_frames_opencv(self, video, start_frame):
        """
        Reads a sequence of frames from a video
        :param video: the video to read the frames from
        :param start_frame: the start frame of the sequence
        :return: a list of frames of length seq_length or None if an error occured
        """
        video_path = os.path.join(self.video_dir, video)
        cap = cv2.VideoCapture(video_path)
        # Seek to start of sequence
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)  # frame 0 = thumbnail of video
        i = 0
        frames = []

        while cap.isOpened() and i < self.seq_length:
            ret, frame = cap.read()
            if not ret:
                print("Error while reading video", video)
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Change to RGB
            frames.append(frame)

            i += 1

        cap.release()

        return frames

    # noinspection PyShadowingNames
    class ToTensor(object):
        """
        Convert nd arrays to tensor

        source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        def __call__(self, sample):
            frames, seq_keypoints, seq_heatmaps = sample['seq'], sample['keypoints'], sample['heatmaps']
            # Swap color axis
            # numpy image H x W x C
            # torch image C x H x W
            for i, image in enumerate(frames):
                image = image.transpose(2, 0, 1)
                frames[i] = torch.from_numpy(image)
                seq_keypoints[i] = torch.from_numpy(seq_keypoints[i])
                seq_heatmaps[i] = torch.from_numpy(seq_heatmaps[i])
            frames = torch.stack(frames)

            sample['seq'], sample['keypoints'], sample['heatmaps'] = frames, seq_keypoints, seq_heatmaps
            return sample

    # noinspection PyShadowingNames
    class Normalize(object):
        """
        Normalize a tensor image with mean and standard deviation.
            Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
            will normalize each channel of the input ``torch.*Tensor`` i.e.
            ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

            .. note::
                This transform acts out of place, i.e., it does not mutates the input tensor.

            Args:
                mean (sequence): Sequence of means for each channel.
                std (sequence): Sequence of standard deviations for each channel.
                inplace(bool,optional): Bool to make this operation in-place.

        source: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Normalize
        """
        def __init__(self, mean, std, inplace=False):
            self.normalize = torchvision.transforms.Normalize(mean, std, inplace)

        def __call__(self, sample):
            frames = sample['seq']
            for i, image in enumerate(frames):
                frames[i] = self.normalize(image)
            sample['seq'] = frames

            return sample

    # noinspection PyShadowingNames
    class Rescale(object):
        """
        Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smallest of image edges is matched
                to output_size keeping aspect ratio the same.

        source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

        def __call__(self, sample):
            frames, seq_keypoints = sample['seq'], sample['keypoints']

            for i, image in enumerate(frames):

                h, w = image.shape[:2]

                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size

                new_h, new_w = int(new_h), int(new_w)

                frames[i] = transform.resize(image, (new_h, new_w))
                # For keypoints we swap h and w because x and y axis are inverted in the image
                seq_keypoints[i] = seq_keypoints[i] * [new_w / w, new_h / h]

            sample['seq'], sample['keypoints'] = frames, seq_keypoints
            return sample

    class CropBody(object):

        def __init__(self, margin=50, crop_width=-1, random=False, square=False, keep_orig=False):
            """
            Crops a sequence of images by placing a bounding box around the keypoints,
            and returns the cropped image and keypoints in a sample

            :param margin: the margin to place around the left, right, top and bottom keypoints. Default=50
            :param crop_width: the width of the bounding box. Leave to -1.
            :param random: whether to add a random margin around the bbox. Default=False.
            :param square: whether the cropping bbox is squared.
            If True, the height and width of the bbox will be set by the
            left-most keypoint and the right-most keypoint + margin. Default=False.
            :param keep_orig: Whether to keep the non-cropped image and corresponding keypoints in the sample.
            Default=False.

            :return: a sample dictionnary containing the sequence of cropped images, and the keypoints.
            If `keep_orig` is set to True, the sample also contains the bboxes, the original (no-cropped) frames, and
            the location of the keypoints in the non-cropped frames.
            """
            self.margin = margin
            self.random = random
            self.square = square
            self.crop_width = crop_width
            self.keep_orig = keep_orig

        def set_crop_width(self, crop_width):
            self.crop_width = crop_width

        def get_crop_width(self):
            return self.crop_width

        def __call__(self, sample):
            frames, seq_keypoints = sample['seq'], sample['keypoints']

            x_min = x_max = y_min = y_max = -1
            crop_height = crop_width = self.crop_width
            bboxes = []
            originals = []
            rnd = None

            # Loop through each frame in the sequence
            for i, image in enumerate(frames):

                if self.keep_orig:
                    originals.append(image)

                keypoints = seq_keypoints[i]

                x_min, y_min = np.nanmin(keypoints, axis=0).astype(int) # Left-most and bottom most keypoints
                x_max, y_max = np.nanmax(keypoints, axis=0).astype(int) # Right-most and top most keypoints

                h, w = image.shape[:2]

                if i == 0 and crop_width == -1:
                    # If it's the first frame in the sequence, and we don't have a defined crop width,
                    # set the width to the distance between the left- and right-most keypoint + twice the margin.
                    # If it's square, the height will be the same as the width
                    crop_height = crop_width = (x_max - x_min) + self.margin * 2
                    if not self.square:
                        # If not square, the height is the distance between the top- and bottom-most keypoint
                        # + twice the margin
                        crop_height = (y_max - y_min) + self.margin * 2

                if i == 0:
                    # define the bbox for the first frame of the sequence,
                    # the other frames in the sequence will have the same bbox.
                    x_mid = x_min + (x_max - x_min) // 2  # horizontal center of bbox
                    x_left = x_mid - crop_width // 2   # left side of bbox

                    if self.random and rnd is None:
                        # add random noise to the bbox
                        rnd = np.random.randint(-self.margin, self.margin)
                        x_left -= rnd

                    # if the bbox is outside of the image to the left, place it at x = 0
                    if x_left < 0:
                        x_left = 0

                    # if the bbox is outside of the image to the right, place it at the border of the image
                    if x_left + crop_width > w:
                        x_left -= (x_left + crop_width) - w

                    x_right = x_left + crop_width  # right side of the bbox

                    y_mid = y_min + (y_max - y_min) // 2  # vertical center
                    y_top = y_mid - crop_height // 2  # top of the bbox

                    if self.random and rnd is None:
                        rnd = np.random.randint(-self.margin, self.margin)
                        y_top -= rnd

                    # if the bbox is outside of the image to the top, place it at y = 0
                    if y_top < 0:
                        y_top = 0

                    # if the bbox is outside of the image to the bottom, place it at the border of the image
                    if y_top + crop_height > h:
                        y_top -= (y_top + crop_height) - h

                    y_bottom = y_top + crop_height  # bottom side of the bbox

                image = image[y_top:y_bottom, x_left:x_right]  # the cropped image

                keypoints = keypoints - [x_left, y_top]   # adjust the keypoints to the cropped image

                frames[i] = image
                seq_keypoints[i] = keypoints

                bboxes.append([y_top, y_bottom, x_left, x_right])

            sample['seq'], sample['keypoints'] = frames, seq_keypoints

            if self.keep_orig:
                sample['bboxes'] = bboxes
                sample['orig_seq'] = originals
                sample['crop_width'] = crop_width

            self.set_crop_width(crop_width)

            return sample

    # noinspection PyShadowingNames
    class RandomCrop(object):
        """
        Randomly crop the sample image

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.

        source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size

        def __call__(self, sample):

            frames, seq_keypoints = sample['seq'], sample['keypoints']
            left, top = -1, -1
            for i, image in enumerate(frames):
                keypoints = seq_keypoints[i]

                h, w = image.shape[:2]
                new_h, new_w = self.output_size

                if new_h < h:
                    if top < 0:
                        top = np.random.randint(0, h - new_h)
                    image = image[top: top + new_h, :]
                    keypoints = keypoints - [0, top]
                    keypoints[keypoints[:, 1] > new_h, ] = None
                    keypoints[keypoints[:, 1] < 0, ] = None

                if new_w < w:
                    if left < 0:
                        left = np.random.randint(0, w - new_w)
                    image = image[:, left: left + new_w]
                    keypoints = keypoints - [left, 0]
                    keypoints[keypoints[:, 0] > new_w, ] = None
                    keypoints[keypoints[:, 0] < 0, ] = None

                frames[i] = image
                seq_keypoints[i] = keypoints

            sample['seq'], sample['keypoints'] = frames, seq_keypoints
            return sample

    # noinspection PyShadowingNames
    class RandomHorizontalFlip(object):
        """Horizontally flip the given image randomly with a given probability.

            Args:
                p (float): probability of the image being flipped. Default value is 0.5

        source; https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html
        """
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, sample):
            frames, seq_keypoints, seq_likelihood = sample['seq'], sample['keypoints'], sample['likelihood']

            if np.random.random() < self.p:
                for i, image in enumerate(frames):
                    frames[i] = np.fliplr(image)
                    w = np.shape(image)[1]
                seq_keypoints[:, :, 0] = w - seq_keypoints[:, :, 0]

                # Swap left and right leg
                left_idx = [0, 1, 2, 6, 7, 8]
                right_idx = [3, 4, 5, 9, 10, 11]

                new_right = seq_keypoints[:, left_idx, :]
                seq_keypoints[:, left_idx, :] = seq_keypoints[:, right_idx, :]
                seq_keypoints[:, right_idx, :] = new_right

            sample['seq'], sample['keypoints'] = frames, seq_keypoints

            return sample

    # noinspection PyShadowingNames
    class RandomRotate(object):
        """
        Rotate the image in a random angle between -theta, +theta.
        Args:
                theta (int, tuple): The range of the rotation angle
        """
        def __init__(self, theta):
            assert isinstance(theta, (int, tuple))
            self.theta = theta

        def __call__(self, sample):
            frames, seq_keypoints = sample['seq'], sample['keypoints']

            # Get a random rotation angle between 0 and the max number of dregrees
            if isinstance(self.theta, int):
                angle = np.random.randint(-self.theta, self.theta)
            else:
                angle = np.random.randint(self.theta[0], self.theta[1])

            # Calculate the rotation matrix
            h, w = np.shape(frames[0])[:2]
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

            for i, image in enumerate(frames):
                # Rotate the image
                frames[i] = cv2.warpAffine(image, rotation_matrix, (w, h))

                # Rotate the keypoints coordinates
                keypoints_rotate = np.transpose(rotation_matrix[:, 0:2])
                keypoints_offset = rotation_matrix[:, 2]
                seq_keypoints[i] = np.matmul(seq_keypoints[i], keypoints_rotate) + keypoints_offset

            sample['seq'], sample['keypoints'] = frames, seq_keypoints
            return sample

    # noinspection PyShadowingNames
    class BrightnessContrast(object):
        """Randomly change the brightness, contrast and saturation of an image.

           Args:
               brightness (tuple of float (min, max)): How much to jitter brightness.
                   Should be between -100 and 100
               contrast (tuple of float (min, max)): How much to jitter contrast.
                   Should be between 0.1 and 3.0

        source: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ColorJitter
       """

        def __init__(self, brightness=(0, 0), contrast=(1, 1)):
            self.brightness = self._check_value(brightness, bound=(-100, 100))
            self.contrast = self._check_value(contrast, bound=(-3.0, 3.0))

        def __call__(self, sample):
            frames = sample['seq']
            brightness_factor = np.random.uniform(self.brightness[0], self.brightness[1])
            contrast_factor = np.random.uniform(self.contrast[0], self.contrast[1])
            for i, image in enumerate(frames):
                new_image = np.zeros_like(image)
                cv2.convertScaleAbs(image, new_image, alpha=contrast_factor, beta=brightness_factor)
                frames[i] = new_image

            sample['seq'] = frames
            return sample

        def _check_value(self, value, bound):
            assert isinstance(value, tuple)
            assert (bound[0] <= value[0] <= value[1] <= bound[1])
            # tuple
            return value


if __name__ == '__main__':
    pass
