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
import yaml
import argparse

class MyConfig(object):

    def __init__(self, args):
        """
         A custom class for reading and parsing a YAML configuration file.

        :param config_path: the path of the configuration file
        """
        config_path = args.config

        self.config = None
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        if self.config is not None:
            # Mandatory parameters
            self.data_folder = self.config['data_folder']
            self.save_checkpoint = self.config['save_checkpoint']
            self.load_checkpoint = self.config['load_checkpoint']

            self.frequent = self.config['frequent']
            self.lr = self.config['lr']
            self.epochs = self.config['epochs']
            self.batch_size = self.config['batch_size']
            self.wandb = self.config['wandb']

            # Optional Parameters
            if 'dataset' in self.config:
                self.dataset = self.config['dataset']

            if 'dataset_csv' in self.config:
                self.dataset_csv = self.config['dataset_csv']

            if 'dataset_test' in self.config:
                self.dataset_test = self.config['dataset_test']

            if 'dataset_h5' in self.config:
                self.dataset_h5 = self.config['dataset_h5']

            if 'regularize' in self.config:
                self.mean = self.config['regularize']['mean']
                self.std = self.config['regularize']['std']
                self.regularize = self.config['regularize']

            if 'lr_decay' in self.config:
                self.lr_decay = self.config['lr_decay']

            if 'seq_length' in self.config:
                self.seq_length = self.config['seq_length']

            if 'group' in self.config:
                self.group = self.config['group']
            else:
                self.group = ""
            if 'optimizer' in self.config:
                self.optimizer = self.config['optimizer']
            else:
                self.optimizer = 'default'
            if 'depth' in self.config:
                self.depth = self.config['depth']
            else:
                self.depth = 3 # initial depth of LEAP

            if 'file_format' in self.config:
                self.file_format = self.config['file_format']

            if 'device' in self.config:
                self.device = self.config['device']
            else:
                self.device = "cpu"

            if 'keypoints' in self.config:
                self.keypoints = self.config['keypoints']

        # override parameters in config file if specified in arguments
        if args.frequent:
            self.frequent = args.frequent
            self.config['frequent'] = args.frequent
        if args.lr:
            self.lr = args.lr
            self.config['lr'] = args.lr
        if args.epochs:
            self.epochs = args.epochs
            self.config['epochs'] = args.epochs
        if args.batch_size:
            self.batch_size = args.batch_size
            self.config['batch_size'] = args.batch_size
        if args.seq_length:
            self.seq_length = args.seq_length
            self.config['seq_length'] = args.seq_length
        if args.optimizer:
            self.optimizer = args.optimizer
            self.config['optimizer'] = args.optimizer
        if args.group:
            self.group = args.group
            self.config['group'] = args.group
        if args.depth:
            self.depth = args.depth
            self.config['depth'] = args.depth

        self.seed = args.seed
        self.original_size = args.original_size

    def __str__(self):
        return str(self.config)


def parse_args(description):
    """
    Parse arguments and process the configuration file
    :return: the config and the arguments
    """
    parser = argparse.ArgumentParser(description=description)
    # config file
    parser.add_argument('--config',
                        help='YAML configuration file',
                        default="cfg/default-config.yml",
                        type=str)
    # training hyper parameters to be overriden from the config file
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate',
                        type=float)
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int)
    parser.add_argument('--batch_size',
                        help='Batch size',
                        type=int)
    parser.add_argument('--seq_length',
                        help='sequence length',
                        type=int)
    parser.add_argument('--seed',
                        help='Random seed',
                        type=int,
                        default=42)
    parser.add_argument('--optimizer',
                        help='type of optimizer',
                        type=str)
    parser.add_argument('--group',
                        help='group name in wandb',
                        type=str)
    parser.add_argument('--depth',
                        help='Depth of LEAP',
                        type=int)

    ###############
    # For testing #
    ###############
    parser.add_argument('--data_root',
                        help='Path to the models',
                        default="~/Datasets",
                        type=str)

    parser.add_argument('--models_path',
                        help='Path to the models',
                        default="",
                        type=str)
    parser.add_argument('--video_dir',
                        help='Path to video',
                        default="",
                        type=str)

    parser.add_argument('--occlusion', dest='occlusion', action='store_true', help='testing with occlusions?')
    parser.add_argument('--no-occlusion', dest='occlusion', action='store_false', help='testing with occlusions?')
    parser.set_defaults(occlusion=False)

    parser.add_argument('--ood', dest='ood', action='store_true', help='testing with ood?')
    parser.add_argument('--no-ood', dest='ood', action='store_false', help='testing with ood?')
    parser.set_defaults(ood=False)

    parser.add_argument('--original_size', dest='original_size', action='store_true', help='use original image size?')
    parser.set_defaults(original_size=False)

    args, rest = parser.parse_known_args()

    print(args)
    cfg = MyConfig(args)
    print(cfg)
    args = parser.parse_args()

    return cfg, args
