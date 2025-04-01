import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


class CompressedDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, resize=None):
        """
        Dataset class for handling compressed data.

        Args:
            data_path (str): Path to the dataset directory.
            train (bool): Whether to load training data or test data.
            transform (callable, optional): Transformations to apply to the data.
            resize (tuple, optional): Resize dimensions for the data.
        """
        self.root_dir = data_path
        self.transform = transform
        self.resize = resize
        self.train = train

        # Load data and labels
        self.data, self.labels = self._load_data()
        self.active_indices = np.arange(len(self.data))

    def _load_data(self):
        """
        Assume with compressed data it can be loaded into memory.
        Load data and labels from the specified directory.
        """
        data_file = "train_data.pt" if self.train else "test_data.pt"
        target_file = "train_target.pt" if self.train else "test_target.pt"

        data_path = os.path.join(self.root_dir, data_file)
        target_path = os.path.join(self.root_dir, target_file)

        with open(data_path, 'rb') as data_file:
            data = torch.load(data_file)

        with open(target_path, 'rb') as target_file:
            labels = torch.load(target_file)

        # Ensure tensors are detached and stacked
        data = torch.stack([tensor.detach() for tensor in data], dim=0)
        labels = torch.tensor(labels)

        return data, labels

    def apply_label_noise(self, noisy_indices):
        """
        Apply random label noise to specified indices.

        Args:
            noisy_indices (list): Indices of labels to be replaced with noise.
        """
        self.noisy_indices = noisy_indices
        self.noisy_initial_labels = self.labels[noisy_indices]
        self.noisy_replacements = []

        possible_labels = np.unique(self.labels)
        for label in self.noisy_initial_labels:
            random_label = label
            while random_label == label:
                random_label = np.random.choice(possible_labels)
            self.noisy_replacements.append(random_label)

        self.labels[noisy_indices] = self.noisy_replacements

    def apply_human_label_noise(self, noise_file_path):
        """
        Apply human label noise from a specified file.

        Args:
            noise_file_path (str): Path to the human noise file.
        """
        noise_file = torch.load(noise_file_path)
        aggre_labels = torch.tensor(noise_file['aggre_label'])
        clean_labels = torch.tensor(noise_file['clean_label'])

        assert torch.equal(clean_labels, self.labels), "Clean labels do not match the dataset labels."

        self.noisy_indices = [idx for idx in range(len(self.labels)) if self.labels[idx] != aggre_labels[idx]]
        self.labels = aggre_labels

        return self.noisy_indices

    def reduce_dataset(self, portion):
        """
        Reduce the dataset to a specified portion.

        Args:
            portion (float): Fraction of the dataset to retain.
        """
        num_samples = int(len(self.labels) * portion)
        self.data = self.data[:num_samples]
        self.labels = self.labels[:num_samples]
        self.active_indices = self.active_indices[:num_samples]

    def filter_by_indices(self, indices):
        """
        Filter the dataset to include only specified indices.

        Args:
            indices (list): Indices to retain in the dataset.
        """
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.active_indices = self.active_indices[indices]

    def remove_index(self, index):
        """
        Remove a specific index from the dataset.

        Args:
            index (int): Index to remove.
        """
        current_idx = self.active_indices.tolist().index(index)
        self.data = np.delete(self.data, current_idx, axis=0)
        self.labels = np.delete(self.labels, current_idx, axis=0)
        self.active_indices = np.delete(self.active_indices, current_idx, axis=0)

    def filter_by_classes(self, classes):
        """
        Filter the dataset to include only specified classes.

        Args:
            classes (list): Classes to retain in the dataset.
        """
        mask = [idx for idx in range(len(self.labels)) if self.labels[idx] in classes]
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.active_indices = self.active_indices[mask]

        # Re-encode labels to be zero-indexed
        label_encoder = preprocessing.LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        self.class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(self.class_mapping)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data, label, idx, additional_info)
        """
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, label, idx, []