import numpy
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        dataset = np.loadtxt('./wine.csv', delimiter=",", dtype=numpy.float32, skiprows=1)
        self.x = torch.from_numpy(dataset[:, 1:])
        self.y = torch.from_numpy(dataset[:, [0]])  # n_samples, 1
        self.n_sample = dataset.shape[0]

    def __getitem__(self, index):
        # get item
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_sample


if __name__ == '__main__': # NB: to run multiprocess dataloader you have to put this at the beginning of the code

    dataset = WineDataset()
    num_epochs = 2
    batch_size = 4
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    total_samples = len(dataset)
    n_iterations = math.ceil(batch_size/4)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader): #l'iterate sul dataloader returna un batch da 4 ogni volta
            if (i+1) % 5 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')




