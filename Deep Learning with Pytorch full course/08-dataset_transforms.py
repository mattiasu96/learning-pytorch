import numpy
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        dataset = np.loadtxt('./wine.csv', delimiter=",", dtype=numpy.float32, skiprows=1)
        self.x = dataset[:, 1:]
        self.y = dataset[:, [0]]  # n_samples, 1
        self.n_sample = dataset.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # get item
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_sample


class ToTensor:  # Transformation da passare alla classe dataset
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        input, target = sample
        input *= self.factor
        return input, target


dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])  # composes multiple transformations
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
