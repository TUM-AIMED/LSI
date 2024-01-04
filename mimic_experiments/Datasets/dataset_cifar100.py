import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing
import sklearn
import cv2
import pickle

class CIFAR100(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False):
        self.root_dir = data_path
        data_files = ["train"]
        test_files = ["test"]
        data = []
        labels = []
        self.gray = False

        if train:
            for file_name in data_files:
                file = os.path.join(data_path, file_name)
                with open(file, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                    data.extend(data_dict [b'data'])
                    labels.extend(data_dict [b'fine_labels'])
        else:
            for file_name in test_files:
                file = os.path.join(data_path, file_name)
                with open(file, 'rb') as fo:
                    data_dict  = pickle.load(fo, encoding='bytes')
                    data.extend(data_dict [b'data'])
                    labels.extend(data_dict [b'fine_labels'])            

        self.data = np.array(data)
        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.labels = np.array(labels)

        self.transform = transform
        self.labels_str = np.unique(self.labels)
        self.active_indices = np.array(range(len(self.data)))
        self.get_normalization()

    def get_normalization(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)])
        transformed_data = np.array([transform(data.transpose(1, 2, 0)).numpy() for data in self.data])

        self.norm = [np.mean(transformed_data[:, i, :, :]) for i in range(transformed_data.shape[1])]
        self.std = [np.std(transformed_data[:, i, :, :]) for i in range(transformed_data.shape[1])]
        
    def reduce_to_active(self, remaining_idx):
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]
    
    def reduce_to_active_class(self, remaining_class):
        mask = [idx for idx in range(len(self.labels)) if self.labels[idx] in remaining_class]
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]

    def remove_index_from_data(self, base_idx):
        current_idx = self.active_indices.tolist().index(base_idx)
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)

    
    def remove_curr_index_from_data(self, idx):
        current_idx = idx
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, normalize=True):
        image = self.data[idx]
        label = self.labels[idx]
        attributes = torch.empty((2,3), dtype=torch.int64)
        image = image.transpose(1, 2, 0)
        if normalize:
            if self.gray:
                transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=3),
                                                transforms.Normalize([0.4874333, 0.4874333, 0.4874333], [0.25059983, 0.25059983, 0.25059983])])
            else:
                transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])])
        else:
            transform = transforms.ToTensor()
        # transform = transforms.ToTensor()
        tensor_image = transform(image)
        tensor_image = tensor_image.to(torch.float32)
        return tensor_image, label, idx, attributes

