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
import torch
from torch import nn
#from torchsummary import summary


class TLEAP(nn.Module):

    def __init__(self, in_channels, out_channels, n_filters=64, depth=3, seq_length=1):
        """
        Implementation of the LEAP model by Pereira et al.
        The Keras implementation can be found at: https://github.com/talmo/leap/
        :param in_channels: input channels (3 for RGB)
        :param out_channels: output channels (corresponds to the number of joints).
        :param n_filters: initial number of filters of the model. It doubles at each new layer in the encoder.
        :param depth: depth of the model. I tested with depth of 3 and 4.
        :param seq_length: length of the sequence; 1 is static (LEAP), 2 is two consecutive frame (T-LEAP), etc.
        With seq_length=1, the model uses 2D convolutions, with seq_length>=2, the model uses 3D convolutions
        """
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.n_filters = n_filters
        self.seq_length = seq_length
        self.is_3D = seq_length > 1

        self._make_layers(depth)

        self.myparams = self.parameters()

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        """
        Creates a convolutional module followed by a ReLU
        """
        if self.is_3D:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=(1, dilation, dilation)),
                nn.ReLU(),
                nn.BatchNorm3d(out_channels)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )

    def _conv_transpose(self, in_channels, out_channels, weight_init=True):
        """
        Creates a transpose convolution module and initialize the weights with Xavier intitialization
        """
        if self.is_3D:
            convT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                       padding=(0, 1, 1), output_padding=(0, 1, 1))
        else:
            convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), output_padding=(1, 1))
        if weight_init:
            convT.weight.data = nn.init.xavier_normal_(convT.weight.data)
        return convT

    def _max_pool(self, kernel_size=2, padding=0):
        if self.is_3D:
            return nn.MaxPool3d(kernel_size=kernel_size, padding=padding)
        else:
            return nn.MaxPool2d(kernel_size=kernel_size, padding=padding)

    def _make_layers(self, depth):
        # Encoder
        enc_sizes = [self.input_channels]
        for i in range(depth):
            enc_sizes.append(self.n_filters * 2**i)

        self.encoder = nn.Sequential()

        for i in range(depth):
            self.encoder.add_module("enc_conv_%d_1" % (i+1), self._conv_block(enc_sizes[i], enc_sizes[i+1], dilation=1))
            self.encoder.add_module("enc_conv_%d_2" % (i+1), self._conv_block(enc_sizes[i+1], enc_sizes[i+1], dilation=1))
            self.encoder.add_module("enc_conv_%d_3" % (i+1), self._conv_block(enc_sizes[i+1], enc_sizes[i+1], dilation=1))
            if i < depth-1:
                if self.is_3D and i == 0:
                    self.encoder.add_module("enc_max_%d" % (i + 1),
                                            self._max_pool(kernel_size=(self.seq_length // 2, 2, 2)))
                elif self.is_3D and i > 1:
                    self.encoder.add_module("enc_max_%d" % (i + 1), self._max_pool(kernel_size=(1, 2, 2)))
                else:
                    self.encoder.add_module("enc_max_%d" % (i+1), self._max_pool())

        # Decoder
        dec_sizes = []
        for i in range(depth-1, 0, -1):
            dec_sizes.append(self.n_filters * 2 ** i)
        self.decoder = nn.Sequential()
        for i in range(len(dec_sizes)-1):
            # Deconv_4
            self.decoder.add_module("dec_convT_%d" % (i+1),
                                    nn.Sequential(self._conv_transpose(dec_sizes[i], dec_sizes[i+1]),
                                                  nn.ReLU()))
            # Conv_5
            self.decoder.add_module("dec_conv_%d_1" % (i+1), self._conv_block(dec_sizes[i+1], dec_sizes[i+1]))
            self.decoder.add_module("dec_conv_%d_2" % (i+1), self._conv_block(dec_sizes[i+1], dec_sizes[i+1]))

        # Deconv_6
        self.decoder.add_module("dec_convT_%d_1" % (depth-1), self._conv_transpose(dec_sizes[-1], dec_sizes[-1]))
        # self.decoder.add_module("dec_conv_%d_2" % (depth-1),
        #                         self._conv_block(dec_sizes[-1], self.output_channels,
        #                                          kernel_size=1, stride=1, padding=0))

        self.fc6 = nn.Linear(dec_sizes[-1], self.output_channels)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, input):
        if len(input.size()) == 5:
            # Input is of size [batch, seq_length, channels, height, width]
            # we want [batch, channels, seq_length, height, width]
            input = input.permute(0, 2, 1, 3, 4).contiguous()

            if not self.is_3D:
                # Get rid of the seq_length dimension when using static leap
                input = input[:, :, -1, :, :]

        out = self.encoder(input)
        out = self.decoder(out)

        if self.is_3D:
            out = out[:, :, -1, :, :]  # Get rid of the seq_length dimension

        out = out.permute([0, 2, 3, 1]).contiguous()  # [batch_size, height, width, channels]
        out = self.fc6(out)
        out = self.softmax(out)  # Softmax on the channel dimension (dim=3)

        # Back to normal dimensions
        out = out.permute([0, 3, 1, 2]).contiguous()  # [batch_size, channels, height, width]

        # If the original size could not be recovered from the deconvolutions
        # Then upsample to original size.
        original_h, original_w = input.size()[-2], input.size()[-1]
        out_h, out_w = out.size()[-2], out.size()[-1]

        if out_h != original_h or out_w != original_w:
            up = nn.Upsample(size=[original_h, original_w], mode='nearest')
            out = up(out)

        return out

    class Permute(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            return x.permute(*self.shape)

if __name__ == '__main__':
    # For debugging only
    # model = TLEAP(in_channels=3, out_channels=17, seq_length=4, depth=4)
    # summary(model, input_data=(4,3,200,200))
    pass
