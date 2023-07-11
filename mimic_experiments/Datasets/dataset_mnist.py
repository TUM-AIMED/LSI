import os
import torch
import numpy as np
from torch.utils.data import Dataset
from mnist import MNIST


class MNISTDataset(Dataset):
    def __init__(self, data_path, indexes=None, train=True, transform=None, values=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.indexes = indexes
        self.values = values
        
        self.images, self.labels = self.load_data()

        # indices of a dataset of just a few classes
        if self.values is not None:
            if self.indexes is not None:
                idx = self.find_indices(self.labels, self.values)
                idx = idx[self.indexes]
                self.images = self.images[idx]
                self.labels = self.labels[idx]
            else:
                self.images = self.images[self.find_indices(self.labels, self.values)]
                self.labels = self.labels[self.find_indices(self.labels, self.values)]


        # active indices of a full dataset
        if self.values is None and self.indexes is not None:
                self.images = self.images[self.indexes]
                self.labels = self.labels[self.indexes]


    def find_indices(self, lst, values):
        if isinstance(values, list):
            return np.array([i for i, x in enumerate(lst) if x in values])
        else:
            return np.array([i for i, x in enumerate(lst) if x == values])
        
    
    def load_data(self):
        mndata = MNIST(self.data_path)
        if self.train:
            images, labels = mndata.load_training()
            images = torch.from_numpy(np.asarray(images)).type(torch.FloatTensor)
            labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
        else:
            images, labels = mndata.load_testing()
            images = torch.from_numpy(np.asarray(images)).type(torch.FloatTensor)
            labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)

        return images, labels
    
    def read_images(self, file):
        _, _, rows, cols = torch.tensor(torch.load(file)).size()
        return torch.tensor(torch.load(file)).reshape(-1, rows, cols).float() / 255.0
    
    def read_labels(self, file):
        return torch.tensor(torch.load(file)).long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx
