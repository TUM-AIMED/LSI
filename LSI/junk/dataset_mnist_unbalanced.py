import os
import torch
import numpy as np
from torch.utils.data import Dataset
from mnist import MNIST

# TODO carry the real indices, to allow for index matching of the gradients and losses
class MNISTDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, classes=None, portions=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.classes = classes
        self.portions = portions
        self.class_assignments = self._generate_class_assignments(classes)
        self.active_indices = np.array(range(len(self.data)))
        
        self.data, self.labels = self.load_data()

        if classes:
            print("Reducing classes")
            self._set_classes(classes)
        if portions and len(portions) != len(self.class_assignments):
            raise Exception("Number of classes and Portions needs to match")
        elif portions and len(portions) == len(self.class_assignments):
            print("Applying portioning")
            self._apply_portions(classes, portions)
            

    def _generate_class_assignments(self, classes):
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
    
    def read_images(self, file):
        _, _, rows, cols = torch.tensor(torch.load(file)).size()
        return torch.tensor(torch.load(file)).reshape(-1, rows, cols).float() / 255.0
    
    def read_labels(self, file):
        return torch.tensor(torch.load(file)).long()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx
