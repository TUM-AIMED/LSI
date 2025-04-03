import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import pickle
import cv2
import pandas as pd
import urllib.request
import tarfile
import zipfile

# Base class for datasets
class BaseDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, shuffle=False, ret4=True):
        """
        Base class for datasets. Handles common functionality like data reduction, label noise, etc.
        """
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.shuffle = shuffle
        self.ret4 = ret4
        self.data = []
        self.labels = []
        self.active_indices = []
        self.attributes = torch.empty((2, 3), dtype=torch.int64)

    def apply_label_noise(self, noisy_idx):
        """
        Apply label noise by replacing labels at specified indices with random labels.
        """
        self.noisy_idx = noisy_idx
        self.noisy_initial_labels = self.labels[self.noisy_idx]
        self.noisy_replacements = []
        possible_labels = np.unique(self.labels)
        for value in self.noisy_initial_labels:
            random_choice = value
            while random_choice == value:
                random_choice = np.random.choice(possible_labels)
            self.noisy_replacements.append(random_choice)
        self.labels[self.noisy_idx] = self.noisy_replacements

    def reduce_to_active(self, remaining_idx):
        """
        Reduce dataset to only include samples at specified indices.
        """
        self.data = self.data[remaining_idx]
        self.labels = self.labels[remaining_idx]
        self.active_indices = self.active_indices[remaining_idx]

    def reduce_to_active_class(self, remaining_class):
        """
        Reduce dataset to only include samples belonging to specified classes.
        """
        mask = [idx for idx in range(len(self.labels)) if self.labels[idx] in remaining_class]
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]

    def remove_index_from_data(self, base_idx):
        """
        Remove a specific index from the dataset.
        """
        current_idx = self.active_indices.tolist().index(base_idx)
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)

    def _set_classes(self, classes):
        """
        Filter dataset to only include specified classes.
        """
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        valid_classes = le.transform(classes)
        mask = np.isin(self.labels, valid_classes)
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]

    def _apply_reduction(self, portions):
        """
        Reduce dataset size based on specified portions.
        """
        num_samples = int(len(self.labels) * portions[0])
        self.data = self.data[:num_samples]
        self.labels = self.labels[:num_samples]
        self.active_indices = self.active_indices[:num_samples]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Abstract method to be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


# CIFAR-10 dataset class
class CIFAR10(BaseDataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize=None, ret4=True, download=True):
        """
        CIFAR-10 dataset loader.
        """
        super().__init__(data_path, train, transform, shuffle, ret4)
        self.resize = resize
        if not os.path.exists(data_path) and download:
            self._download_data()
        self._load_data()
        if classes is not None:
            self._set_classes(classes)
        if portions is not None:
            self._apply_reduction(portions)

    def _download_data(self):
        """
        Download CIFAR-10 dataset from public domain.
        """
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        archive_path = os.path.join(self.data_path, "cifar-10-python.tar.gz")
        os.makedirs(self.data_path, exist_ok=True)
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_path)

    def _load_data(self):
        """
        Load CIFAR-10 data from binary files.
        """
        data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_files = ["test_batch"]
        files = data_files if self.train else test_files

        for file_name in files:
            file = os.path.join(self.data_path, "cifar-10-batches-py", file_name)
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                self.data.extend(data_dict[b'data'])
                self.labels.extend(data_dict[b'labels'])

        self.data = np.array(self.data).reshape((-1, 3, 32, 32))
        self.labels = np.array(self.labels)
        self.active_indices = np.arange(len(self.data))

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, self.attributes


# CIFAR-100 dataset class
class CIFAR100(BaseDataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize=None, ret4=True, download=True):
        """
        CIFAR-100 dataset loader.
        """
        super().__init__(data_path, train, transform, shuffle, ret4)
        self.resize = resize
        if not os.path.exists(data_path) and download:
            self._download_data()
        self._load_data()
        if classes is not None:
            self._set_classes(classes)
        if portions is not None:
            self._apply_reduction(portions)

    def _download_data(self):
        """
        Download CIFAR-100 dataset from public domain.
        """
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        archive_path = os.path.join(self.data_path, "cifar-100-python.tar.gz")
        os.makedirs(self.data_path, exist_ok=True)
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_path)

    def _load_data(self):
        """
        Load CIFAR-100 data from binary files.
        """
        data_files = ["train"]
        test_files = ["test"]
        files = data_files if self.train else test_files

        for file_name in files:
            file = os.path.join(self.data_path, "cifar-100-python", file_name)
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                self.data.extend(data_dict[b'data'])
                self.labels.extend(data_dict[b'fine_labels'])

        self.data = np.array(self.data).reshape((-1, 3, 32, 32))
        self.labels = np.array(self.labels)
        self.active_indices = np.arange(len(self.data))

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, self.attributes


# ImageNet dataset class
class Imagenet(BaseDataset):
    def __init__(self, data_path, train=True, transform=None, shuffle=False, download=False):
        """
        ImageNet dataset loader.
        """
        super().__init__(data_path, train, transform, shuffle)
        if not os.path.exists(data_path) and download:
            raise ValueError("ImageNet dataset is not publicly available for download. Please provide the dataset manually.")
        self._load_data()

    def _load_data(self):
        """
        Load ImageNet data from directory structure.
        """
        dataset_type = "train" if self.train else "val"
        self.data, self.labels = self._get_image_paths_and_labels(self.data_path, dataset_type)
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.active_indices = np.arange(len(self.data))

    def _get_image_paths_and_labels(self, base_path, dataset_type):
        """
        Get image paths and labels from directory structure.
        """
        image_paths = []
        labels = []
        dataset_path = os.path.join(base_path, dataset_type)

        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(label_path, image_file))
                        labels.append(label)

        return image_paths, labels

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, self.attributes


# Imagenette dataset class
class ImagenetteDataset(BaseDataset):
    def __init__(self, data_path, train=True, transform=None, shuffle=False, ret4=True, smaller=False, download=False):
        """
        Imagenette dataset loader.
        """
        super().__init__(data_path, train, transform, shuffle, ret4)
        self.smaller = smaller
        if not os.path.exists(data_path) and download:
            self._download_data()
        self._load_data()

    def _download_data(self):
        """
        Download Imagenette dataset from public domain.
        """
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        archive_path = os.path.join(self.data_path, "imagenette2-160.tgz")
        os.makedirs(self.data_path, exist_ok=True)
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_path)

    def _load_data(self):
        """
        Load Imagenette data from CSV file.
        """
        csv_file = os.path.join(self.data_path, "source_labels.csv")
        df = pd.read_csv(csv_file)

        # Filter data based on train/validation split
        mask = df["in_val"] == (0 if self.train else 1)
        filtered_df = df[mask]

        self.data = filtered_df["Path"].values
        self.labels = filtered_df["encoded_labels"].values
        self.active_indices = np.arange(len(self.data))

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        img_path = os.path.join(self.data_path, self.data[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, self.attributes


# Prima dataset class
class Prima(BaseDataset):
    def __init__(self, data_path, train=True, transform=None, shuffle=False, ret4=True, smaller=False, download=False):
        """
        Prima dataset loader for pneumonia classification.
        """
        super().__init__(data_path, train, transform, shuffle, ret4)
        self.smaller = smaller
        if not os.path.exists(data_path) and download:
            raise ValueError("Prima dataset is not publicly available for download. Please provide the dataset manually.")
        self._load_data()

    def _load_data(self):
        """
        Load Prima data from directory structure.
        """
        data_path = os.path.join(self.data_path, 'train')
        categories = ["bacterial_pneumonia", "normal", "viral_pneumonia"]

        data, labels = [], []
        for i, category in enumerate(categories):
            category_path = os.path.join(data_path, category)
            img_paths = [os.path.join(category_path, img) for img in os.listdir(category_path)]
            data.extend(img_paths)
            labels.extend([i] * len(img_paths))

        split_data = train_test_split(data, labels, test_size=0.1, random_state=42)
        self.data, self.labels = (split_data[0], split_data[2]) if self.train else (split_data[1], split_data[3])

        self.active_indices = np.arange(len(self.data))
        self.data = torch.stack([self._process_image(path) for path in tqdm(self.data)])
        self.labels = torch.tensor(self.labels)

    def _process_image(self, path):
        """
        Process and resize an image.
        """
        size = (128, 128) if self.smaller else (512, 512)
        image = Image.open(path).convert('L').convert("RGB").resize(size)
        return transforms.ToTensor()(image)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        image = self.data[idx]
        label = self.labels[idx]
        return image, label, idx, self.attributes
