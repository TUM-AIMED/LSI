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
from tqdm import tqdm

class Imagenet(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False):
        self.root_dir = data_path
        data_files = ["train"]
        test_files = ["test"]
        data = []
        labels = []
        self.gray = False

        data_test, labels_test = self.get_image_paths_and_labels(data_path, "val")
        if train:
            data, labels = self.get_image_paths_and_labels(data_path, "train")
        else:
            data, labels = data_test, labels_test
  

        self.data = np.array(data)
        le = preprocessing.LabelEncoder()
        le.fit(labels_test)
        self.labels = le.transform(labels)

        self.labels_str = np.unique(self.labels)
        self.active_indices = np.array(range(len(self.data)))
        self.get_normalization()

    def get_image_paths_and_labels(self, base_path, dataset_type):
        image_paths = []
        labels = []

        # Construct the path to the specified dataset (train or val)
        dataset_path = os.path.join(base_path, dataset_type)

        # Iterate over class labels
        for label in tqdm(os.listdir(dataset_path)):
            label_path = os.path.join(dataset_path, label)

            # Check if the item is a directory (avoid hidden files, etc.)
            if os.path.isdir(label_path):
                # Iterate over images in the class label folder
                for image_file in os.listdir(label_path):
                    # Ensure the file is an image (you can customize this condition)
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Construct the full path to the image
                        image_path = os.path.join(label_path, image_file)

                        # Append the image path and corresponding label to the lists
                        image_paths.append(image_path)
                        labels.append(label)

        return image_paths, labels
    
    def get_normalization(self):
        self.norm = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
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


    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, normalize=True):
        img_name = self.data[idx]
        image = self.pil_loader(img_name)
        label = self.labels[idx]
        attributes = torch.empty((2,3), dtype=torch.int64)
        if normalize:
            if self.gray:
                transform = transforms.Compose([transforms.Resize(256),           # Resize the image to 256x256 pixels
                                                transforms.CenterCrop(224),  
                                                transforms.ToTensor(), 
                                                transforms.Grayscale(num_output_channels=3),
                                                transforms.Normalize([0.4874333, 0.4874333, 0.4874333], [0.25059983, 0.25059983, 0.25059983])])
            else:
                transform = transforms.Compose([
                            transforms.Resize(256),           # Resize the image to 256x256 pixels
                            transforms.CenterCrop(224),  
                            transforms.ToTensor(),
                            transforms.Normalize(self.norm, self.std)])
        else:
            transform = transforms.ToTensor()
        # transform = transforms.ToTensor()
        tensor_image = transform(image)
        tensor_image = tensor_image.to(torch.float32)
        return tensor_image, label, idx, attributes



# dataset = Imagenet("/vol/aimspace/projects/ILSVRC2012/", train=False)
# ex = dataset[0]
# print("")