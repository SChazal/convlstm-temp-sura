import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from sklearn import preprocessing


def load_ecmwf(root):
    # Load MNIST dataset for generating training data.
    npdata = np.load(root)['arr_0']
    return npdata


class ClimateData_MC(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, target, transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(ClimateData_MC, self).__init__()

        self.dataset = None
        self.ecmwf = load_ecmwf(root)

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.valid_length = 10
        self.length = self.ecmwf.shape[0] - \
            self.n_frames_total - self.valid_length
        if not is_train:
            self.length = self.valid_length
        self.train_length = self.ecmwf.shape[0] - \
            self.n_frames_total - self.valid_length
        self.transform = transform
        # For generating data
        self.image_height = 73
        self.image_width = 144
        self.image_input_channels = self.ecmwf[0].shape[0]
        self.target = target

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train:
            images = self.ecmwf[idx:  (idx + length), ...]
        else:
            images = self.ecmwf[self.train_length +
                                idx: (self.train_length + idx + length), ...]

        # r = 1
        # w = int(64 / r)
        # images = images.reshape((length, w, r, w, r)).transpose(
        #     0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[: self.n_frames_input]
        if self.n_frames_output > 0:
            # output only temperature
            output = images[self.n_frames_input:length]
            # output = images[self.n_frames_input:length]
        else:
            output = []
        frozen = input[-1]
        output = torch.from_numpy(output/1000).contiguous()
        input = torch.from_numpy(input/1000).contiguous()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length
