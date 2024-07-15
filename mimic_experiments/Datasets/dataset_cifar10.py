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


class CIFAR10(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None, ret4=True):
        self.ret4 = ret4
        self.root_dir = data_path
        self.resize = False
        if resize != None:
            self.resize = resize
        data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_files = ["test_batch"]
        data = []
        labels = []

        if train:
            for file_name in data_files:
                file = os.path.join(data_path, file_name)
                with open(file, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                    data.extend(data_dict [b'data'])
                    labels.extend(data_dict [b'labels'])
        else:
            for file_name in test_files:
                file = os.path.join(data_path, file_name)
                with open(file, 'rb') as fo:
                    data_dict  = pickle.load(fo, encoding='bytes')
                    data.extend(data_dict [b'data'])
                    labels.extend(data_dict [b'labels'])            

        self.data = np.array(data)
        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.labels = np.array(labels)
        self.attributes = torch.empty((2,3), dtype=torch.int64)
        if not ret4:
            self.transform_data()
        self.transform = transform
        self.labels_str = np.unique(self.labels)
        self.active_indices = np.array(range(len(self.data)))
        
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

        print(dict(zip(le.classes_, le.transform(le.classes_))))
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

    def apply_image_mark(self, noisy_idx):
        self.noisy_idx = noisy_idx
        for data in self.data[self.noisy_idx]:
            square_size = random.randint(6, 10)
            x_pos = random.randint(0, 32 - square_size)
            y_pos = random.randint(0, 32 - square_size)
            data[:, x_pos:x_pos + square_size, y_pos:y_pos + square_size] = 0
        return


    def _set_classes(self, classes):
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.class_assignments = le
        valid_classes = self.class_assignments.transform(classes)
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
        
    def reduce_to_active_class(self, remaining_class):
        mask = [idx for idx in range(len(self.labels)) if self.labels[idx] in remaining_class]
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]
        
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

    def batchwise_reorder(self, batchsize, firstbatchnum, remove=False):
        start = firstbatchnum * batchsize
        end = start + batchsize
        if remove:
            end = end-1
        self.data = np.concatenate((self.data[start:end], self.data[:start], self.data[end:]))
        self.labels = np.concatenate((self.labels[start:end], self.labels[:start], self.labels[end:]))
        self.active_indices = np.concatenate((self.active_indices[start:end], self.active_indices[:start], self.active_indices[end:]))

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = image.transpose(1, 2, 0)
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        # tensor_image = tensor_image[0, :, :] 
        
        return tensor_image, label, idx, self.attributes


    def transform_data(self):
        data_new = []
        for i, data in enumerate(self.data):
            image = data
            image = image.transpose(1, 2, 0)
            transform = transforms.ToTensor()
            transform2 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) 
            tensor_image = transform(image)
            tensor_image = transform2(tensor_image)
            data_new.append(tensor_image)
        # data_new = [data for lab, data in zip(self.labels, data_new) if lab in [0, 1, 2, 3]]
        # labels_new = [lab for lab in self.labels if lab in [0, 1, 2, 3]]
        # self.data = torch.stack(data_new) 
        # self.labels = torch.tensor(labels_new)
        self.data = torch.stack(data_new)

