import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        self.labels = self.df[self.label_class].values
        self.class_assignments = self._generate_class_assignments(classes)
        self.active_indices = np.array(range(len(self.data)))
        
        if classes:
            print("Reducing classes")
            self._set_classes(classes)
        if portions and len(portions) != len(self.class_assignments):
            raise Exception("Number of classes and Portions needs to match")
        elif portions and len(portions) == len(self.class_assignments):
            print("Applying portioning")
            self._apply_portions(classes, portions)
        self.labels =[classes.index(label) for label in self.labels]
        self.labels = torch.tensor(self.labels)

    def _generate_class_assignments(self, classes):
        unique_classes = np.unique(self.labels)
        class_assignments = {label: classes.index(label) for label in classes}
        return class_assignments

    def _set_classes(self, classes):
        valid_classes = [class_label for class_label in classes if class_label in self.class_assignments]
        if len(valid_classes) == 0:
            raise ValueError("No valid classes found in the dataset.")
        if len(valid_classes) != len(classes):
            raise ValueError("Class not found")
        self.class_assignments = {class_label: class_label for class_label in valid_classes}
        remaining_idx = [idx for idx, label in enumerate(self.labels) if label in classes]
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

    def _apply_portions(self, classes, portions):
        remaining_idx = []
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

    def reduce_to_active(self, remaining_idx):
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        
        return tensor_image, label, idx
