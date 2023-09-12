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

class CIFAR01(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False):
        self.root_dir = data_path
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


    def _set_classes(self, classes):
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.class_assignments1 = le
        print(dict(zip(le.classes_, le.transform(le.classes_))))
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

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=0)
        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        attributes = torch.empty((2,3), dtype=torch.int64)
        
        return tensor_image, label, idx, attributes

