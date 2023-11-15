import os
import sys
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing
import sklearn
import cv2
from mnist import MNIST
import random

class MNISTDataset(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None):
        print("Loading data", flush=True)
        self.data_path = data_path
        self.train = train
        self.resize = resize
        self.transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        
        self.data, self.labels = self.load_data()
        self.data = self.data.reshape((self.data.shape[0], 1, 28, 28))
        test = np.array(self.data)
        # self.data = self.data.repeat(1, 3, 1, 1)
        self.data = self.data/255
        self.labels = np.array(self.labels)
        self.data = np.array(self.data)

        self.labels_str = np.unique(self.labels)
        self.active_indices = np.array(range(len(self.data)))
        print("Loading data2", flush=True)
        if portions == None:
            len_portions = 0
        else:
            len_portions = len(portions)
        if classes == None:
            len_classes = 0
        else:
            len_classes = len(classes)
        
        if len_classes !=  0 and len_portions == len_classes:
            print("Reducing classes")
            self._set_classes(classes)
        if len_portions == 0:
            print("No Portioning")
        elif len_portions == 1 and len_classes == 0:
            print("Reducing size of Dataset")
            self._apply_reduction(portions)
        elif len_portions != len_classes:
            if len_portions == 1:
                self._apply_reduction(portions)
            else:
                raise Exception("Number of classes and Portions needs to match")
        elif portions != 0 and len_portions == len_classes:
            print("Applying Class-specific Portioning")
            self._apply_portions(classes, portions)

        if shuffle:
            self.data, self.labels, self.active_indices = sklearn.utils.shuffle(
                self.data, 
                self.labels, 
                self.active_indices,
                random_state=1)
        
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.class_assignments2 = le
        

    def _set_classes(self, classes):
        pot_classes = np.unique(self.labels)
        valid_classes = [cl for cl in classes if cl in pot_classes]
        if len(valid_classes) == 0:
            raise ValueError("No valid classes found in the dataset.")
        if len(valid_classes) != len(classes):
            raise ValueError("Class not found")
        
        remaining_idx = [idx for idx, label in enumerate(self.labels) if label in valid_classes]
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]
        le2 = preprocessing.LabelEncoder()
        le2.fit(self.labels)
        self.labels = le2.transform(self.labels)


    def _apply_portions(self, classes, portions):
        remaining_idx = []
        class_sizes = []
        classes = self.class_assignments.transform(classes)
        for class_label, portion in zip(classes, portions):
            class_idx = [idx for idx, label in enumerate(self.labels) if label == class_label]
            class_sizes.append(len(class_idx))
        for class_label, portion in zip(classes, portions):
            class_idx = [idx for idx, label in enumerate(self.labels) if label == class_label]
            num_samples = int(min(class_sizes) * portion)
            sampled_data = class_idx[:num_samples]
            remaining_idx.extend(sampled_data)
        remaining_idx.sort()
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

    def _apply_reduction(self, portions):
        num_samples = int(len(self.labels) * portions[0])
        self.data = self.data[:num_samples]
        self.labels = self.labels[:num_samples]
        self.active_indices = self.active_indices[:num_samples]

    def reduce_to_active(self, remaining_idx):
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

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

    def zero_index_from_data(self, base_idx):
        current_idx = self.active_indices.tolist().index(base_idx)
        self.data[current_idx] = np.zeros(self.data[0].shape)
        self.labels = np.random.randint(10)

    def batchwise_reorder(self, batchsize, firstbatchnum, remove=False):
        start = firstbatchnum * batchsize
        end = start + batchsize
        if remove:
            end = end-1
        self.data = np.concatenate((self.data[start:end], self.data[:start], self.data[end:]))
        self.labels = np.concatenate((self.labels[start:end], self.labels[:start], self.labels[end:]))
        self.active_indices = np.concatenate((self.active_indices[start:end], self.active_indices[:start], self.active_indices[end:]))
        
    
    def load_data(self):
        print("Pre true load", flush=True)
        mndata = MNIST(self.data_path)
        print("After true load", flush=True)
        if self.train:
            images, labels = mndata.load_training()
            images = torch.from_numpy(np.asarray(images)).type(torch.FloatTensor)
            labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
        else:
            images, labels = mndata.load_testing()
            images = torch.from_numpy(np.asarray(images)).type(torch.FloatTensor)
            labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
        print("After after true load", flush=True)
        return images, labels

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

    def apply_image_mark(self, noisy_idx):
        self.noisy_idx = noisy_idx
        for data in self.data[self.noisy_idx]:
            square_size = random.randint(6, 10)
            x_pos = random.randint(0, 32 - square_size)
            y_pos = random.randint(0, 32 - square_size)
            data[:, x_pos:x_pos + square_size, y_pos:y_pos + square_size] = 0
        return
    

    def apply_group_label_noise(self, lab=6, nth=1, under=400):
        noisy_idx = []
        possible_labels = np.unique(self.labels)
        for idx, label in enumerate(self.labels):
            if label == lab and idx % nth == 0 and idx < 400:
                self.labels[idx] ==  np.random.choice(possible_labels)
                noisy_idx.append(idx)
        return noisy_idx



    def read_images(self, file):
        _, _, rows, cols = torch.tensor(torch.load(file)).size()
        return torch.tensor(torch.load(file)).reshape(-1, rows, cols).float() / 255.0
    
    def read_labels(self, file):
        return torch.tensor(torch.load(file)).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        # if len(image.shape) == 2:
        #     image = np.stack([image] * 3, axis=0)
        # if self.resize:
        #     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor_image = torch.from_numpy(image)
        attributes = torch.empty((2,3), dtype=torch.int64)
        
        return tensor_image, label, idx, attributes