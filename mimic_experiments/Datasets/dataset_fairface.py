import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing

class FairfaceDataset(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None):
        self.label_class = "race"
        self.root_dir = data_path
        if train:
            self.csv_file = "fairface_label_train.csv"
        else:
            self.csv_file = "fairface_label_val.csv"

        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_path, self.csv_file))
        self.data = self.df["file"].values
        self.labels_str = self.df[self.label_class].values
        self.attributes = np.array(self.df[["age", "gender", "race"]].values)
        self.labels = self.df[self.label_class].values
        self.active_indices = np.array(range(len(self.data)))

        self.attribute_encoder = []

        for index in range(self.attributes.shape[1]):
            le = preprocessing.LabelEncoder()
            le.fit(self.attributes[:, index])
            self.attributes[:, index] = le.transform(self.attributes[:, index])
            self.attribute_encoder.append(le)

        
        if classes:
            print("Reducing classes")
            self._set_classes(classes)
        if portions and len(portions) != len(classes):
            raise Exception("Number of classes and Portions needs to match")
        elif portions and len(portions) == len(classes):
            print("Applying portioning")
            self._apply_portions(classes, portions)
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.class_assignments2 = le


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
        self.attributes = self.attributes[remaining_idx]

    def _apply_portions(self, classes, portions):
        remaining_idx = []
        classes = self.class_assignments.transform(classes)
        for class_label, portion in zip(classes, portions):
            class_idx = [idx for idx, label in enumerate(self.labels) if label == class_label]
            class_size = len(class_idx)
            num_samples = int(class_size * portion)
            sampled_data = class_idx[:num_samples]
            remaining_idx.extend(sampled_data)
        remaining_idx.sort()
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]
        self.attributes = self.attributes[remaining_idx]

    def reduce_to_active(self, remaining_idx):
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]
        self.attributes = self.attributes[remaining_idx]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        attributes = self.attributes[idx]
        
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        
        return tensor_image, label, idx, torch.tensor(attributes.astype(int))
