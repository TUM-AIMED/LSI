import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing
import sklearn

class ImagenetteDataset(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, ret4=True, smaller=False):
        self.smaller = smaller
        self.label_class = "encoded_labels"
        self.root_dir = data_path
        self.csv_file = "source_labels.csv"

        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_path, self.csv_file))

        self.data = self.df["Path"].values
        self.labels = self.df[self.label_class].values
        self.inval = self.df["in_val"].values
        
        if train:
            self.data = [data for in_val, data in zip(self.inval, self.data) if in_val == 0]
            self.labels = [lab for in_val, lab in zip(self.inval, self.labels) if in_val == 0]
        else:
            self.data = [data for in_val, data in zip(self.inval, self.data) if in_val == 1]
            self.labels = [lab for in_val, lab in zip(self.inval, self.labels) if in_val == 1]

        self.active_indices = np.array(range(len(self.data)))

        if not ret4:
            self.transform_data()
        self.labels = torch.tensor(self.labels)
        
        if shuffle:
            self.data, self.labels, self.attributes, self.active_indices = sklearn.utils.shuffle(
                self.data, 
                self.labels, 
                self.attributes, 
                self.active_indices,
                random_state=1)


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
        
        transform = transforms.ToTensor()
        transform2 = transforms.Resize((160, 160))
        tensor_image = transform(image)
        tensor_image = transform2(tensor_image)
        if tensor_image.shape[0] == 1:
            tensor_image = torch.cat((tensor_image, tensor_image,tensor_image), dim=0)
        
        return tensor_image, label

    def transform_data(self):
        data_new = []
        for i, data in enumerate(self.data):
            image = Image.open(data)
            transform = transforms.ToTensor()
            if not self.smaller:
                transform2 = transforms.Resize((160, 160))
            else:
                transform2 = transforms.Resize((100, 100))
            tensor_image = transform(image)
            tensor_image = transform2(tensor_image)
            if tensor_image.shape[0] == 1:
                tensor_image = torch.cat((tensor_image, tensor_image,tensor_image), dim=0)
            data_new.append(tensor_image)
        # data_new = [data for lab, data in zip(self.labels, data_new) if lab in [0, 1, 2, 3]]
        # labels_new = [lab for lab in self.labels if lab in [0, 1, 2, 3]]
        # self.data = torch.stack(data_new) 
        # self.labels = torch.tensor(labels_new)
        self.data = torch.stack(data_new)