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
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


class Prima(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None, ret4=True, smaller=False):
        self.root_dir = data_path
        self.resize = False
        data_path = os.path.join(self.root_dir, 'train')
        data_path1 = os.path.join(data_path, "bacterial_pneumonia")
        data_path2 = os.path.join(data_path, "normal")
        data_path3 = os.path.join(data_path, "viral_pneumonia")

        img_paths1 = os.listdir(data_path1)
        img_paths2 = os.listdir(data_path2)
        img_paths3 = os.listdir(data_path3)

        img_paths1 = [os.path.join(data_path1, img_path) for img_path in img_paths1]
        img_paths2 = [os.path.join(data_path2, img_path) for img_path in img_paths2]
        img_paths3 = [os.path.join(data_path3, img_path) for img_path in img_paths3]


        n_classes1 = [0] * len(img_paths1)
        n_classes2 = [1] * len(img_paths2)
        n_classes3 = [2] * len(img_paths3)

        data = img_paths1 + img_paths2 + img_paths3
        labels = n_classes1 + n_classes2 + n_classes3

        random.seed(42)
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data, labels = zip(*combined)
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.1, random_state=42)
           



        if train:
            self.data = data_train
            self.labels = labels_train
        else:
            self.data = data_val
            self.labels = labels_val
        self.active_indices = np.array([i for i in range(len(self.data))])
        result_data = []
        for path in tqdm(self.data):
            if not smaller:
                image = Image.open(path).convert('L').convert("RGB").resize((512, 512))
            else:
                image = Image.open(path).convert('L').convert("RGB").resize((128, 128))
            result_data.append(pil_to_tensor(image) / 255.0)  
        self.data = torch.stack(result_data)
        self.labels = torch.tensor(self.labels)

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
        data = self.data[idx]
        label = self.labels[idx]      
        return data, label, idx, []