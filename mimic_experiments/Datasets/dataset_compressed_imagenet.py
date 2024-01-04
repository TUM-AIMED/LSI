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
import random
from tqdm import tqdm


class Imagenet(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None):
        self.root_dir = data_path
        self.resize = False

        if train:
            data_path = os.path.join(self.root_dir, "train_data2")
            targets_paths = data_path +  "/targets/target.pt"
            with open(targets_paths, 'rb') as fo:
                indiv_data = torch.load(fo)
                labels = indiv_data[0]
                dataidx = indiv_data[1]
            data_paths = [data_path +  "/data/" + str(idx) + ".pt" for idx in dataidx]
            print("")
        else:
            data_path = os.path.join(self.root_dir, "test_data2")
            targets_paths = data_path +  "/targets/target.pt"
            with open(targets_paths, 'rb') as fo:
                indiv_data = torch.load(fo)
                labels = indiv_data[0]
                dataidx = indiv_data[1]
            data_paths = [data_path +  "/data/" + str(idx) + ".pt" for idx in dataidx]
        
        self.data=data_paths
        self.labels=labels
        self.active_indices=dataidx

        seed_value = 42
        random.seed(seed_value)

        combined_lists = list(zip(self.data, self.labels, self.active_indices))
        random.shuffle(combined_lists)
        self.data, self.labels, self.active_indices = zip(*combined_lists)


    def apply_label_noise(self, noisy_idx):
        self.noisy_idx = noisy_idx
        self.noisy_initial_labels = self.labels[self.noisy_idx]
        self.noisy_replacements = []
        possible_labels = np.unique(self.labels)
        for i, value in enumerate(self.noisy_initial_labels):
            random_choice = value  # Initialize with the specific value
            while random_choice == value:
                random_choice = np.random.choice(possible_labels)
            self.noisy_replacements.append(random_choice)
        
        self.labels[self.noisy_idx] = self.noisy_replacements
        return

    def _apply_reduction(self, portions):
        num_samples = int(len(self.labels) * portions[0])
        self.data = self.data[:num_samples]
        self.labels = self.labels[:num_samples]
        self.active_indices = self.active_indices[:num_samples]

    def reduce_to_active(self, remaining_idx):
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

    def remove_curr_index_from_data(self, idx):
        current_idx = idx
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)

    def remove_index_from_data(self, base_idx):
        current_idx = self.active_indices.tolist().index(base_idx)
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)
        
    def reduce_to_active_class(self, remaining_class):
        mask = [idx for idx in range(len(self.labels)) if self.labels[idx] in remaining_class]
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.class_assignments2 = le
        print(dict(zip(le.classes_, le.transform(le.classes_))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        with open(path, 'rb') as fo:
            data = torch.load(fo)
        label = self.labels[idx]        
        return data, label, idx, []
    
path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Imagenet"
dataset = Imagenet(path, train=False)
x1 = dataset[0]