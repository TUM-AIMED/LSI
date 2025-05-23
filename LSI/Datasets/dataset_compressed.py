import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import random  # Added for corrupt_label and corrupt_data
from tqdm import tqdm
import requests

class CompressedDataset(Dataset):
    def __init__(self, root_dir=None, train=True, transform=None, resize=None, data=None, labels=None):
        """
        Dataset class for handling compressed data.

        Args:
            root_dir (str, optional): Directory containing the dataset files.
            train (bool, optional): Whether to load training data or test data.
            transform (callable, optional): Transformations to apply to the data.
            resize (tuple, optional): Resize dimensions for the data.
            data (torch.Tensor, optional): Preloaded data tensor.
            labels (torch.Tensor, optional): Preloaded labels tensor.
        """
        self.transform = transform
        self.resize = resize

        if data is not None and labels is not None:
            # Initialize with provided data and labels
            self.data = data
            self.labels = labels
            self.active_indices = np.arange(len(self.data))
        elif root_dir is not None:
            # Initialize by loading data from the specified directory
            self.root_dir = root_dir
            self.train = train
            self.data, self.labels = self._load_data()
            self.active_indices = np.arange(len(self.data))
        else:
            raise ValueError("Either root_dir or both data and labels must be provided.")

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

    def corrupt_label(self, corrupt):
        """
        Corrupt a portion of the labels in the dataset.

        Args:
            corrupt (float): Fraction of labels to corrupt.

        Returns:
            list: Indices of corrupted labels.
        """
        y_unique = torch.unique(self.labels)
        idx_list = list(range(len(self.labels)))
        corrupted_idx = random.sample(idx_list, int(corrupt * len(self.labels)))
        for idx in corrupted_idx:
            org_class = self.labels[idx]
            sampled_class = random.randint(0, len(y_unique) - 1)
            while sampled_class == org_class:
                sampled_class = random.randint(0, len(y_unique) - 1)
            self.labels[idx] = sampled_class
        return corrupted_idx

    def corrupt_data(self, args, noise_level, corrupt_data_label):
        """
        Corrupt a portion of the data in the dataset.

        Args:
            args: Arguments containing dataset information.
            noise_level (float): Level of noise to add to the data.
            corrupt_data_label (int): Label of data to corrupt.

        Returns:
            list: Indices of corrupted data.
        """
        if args.dataset == "Imdbcompressed":
            n_corrupt = int(len(self.data) * noise_level)
            lorem_data_set_class, lorem_data_path = get_dataset("Loremcompressed")
            lorem_dataset = lorem_data_set_class(lorem_data_path, train=True)
            X_corrupt = lorem_dataset.data[:n_corrupt]
            y_corrupt = torch.cat((torch.ones(n_corrupt // 2, dtype=torch.int), torch.zeros(n_corrupt // 2, dtype=torch.int)))
            y_corrupt = y_corrupt[torch.randperm(n_corrupt)]
            self.data = torch.concat([self.data, X_corrupt])
            self.labels = torch.concat([self.labels, y_corrupt])
            corrupted_idx = list(range(len(self.data) - n_corrupt, len(self.data)))
            return corrupted_idx
        else:
            corrupted_idx = []
            std = torch.std(self.data)
            for i, (data_indiv, label_indiv) in enumerate(zip(self.data, self.labels)):
                if label_indiv in corrupt_data_label:
                    self.data[i] += std * noise_level * torch.randn(data_indiv.shape)
                    corrupted_idx.append(i)
            return corrupted_idx

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

    def apply_human_label_noise(self, noise_file_path, dname):
        """
        Apply human label noise from a specified file.

        Args:
            noise_file_path (str): Path to the human noise file.
            dname (str): Dataset name, either 'cifar10' or 'cifar100'.
        """
        if dname in ["cifar10", "cifar10compressed", "cifar10_compressed"]:
            url = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
            label_key = 'aggre_label'
            noise_file_path = os.path.join(noise_file_path, "CIFAR-10_human.pt")
        elif dname in ["cifar100", "cifar100compressed", "cifar100_compressed"]:
            url = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt"
            label_key = 'noisy_label'
            noise_file_path = os.path.join(noise_file_path, "CIFAR-100_human.pt")
        else:
            raise ValueError("Unsupported dataset name. Use 'cifar10' or 'cifar100'.")

        if not os.path.exists(noise_file_path):
            os.makedirs(os.path.dirname(noise_file_path), exist_ok=True)
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(noise_file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded and saved noise file to {noise_file_path}")
            else:
                raise RuntimeError(f"Failed to download the noise file from {url}. HTTP status code: {response.status_code}")

        noise_file = torch.load(noise_file_path, weights_only=False)
        noisy_labels = torch.tensor(noise_file[label_key])
        clean_labels = torch.tensor(noise_file['clean_label'])

        assert torch.equal(clean_labels, self.labels), "Clean labels do not match the dataset labels."

        self.noisy_indices = [idx for idx in range(len(self.labels)) if self.labels[idx] != noisy_labels[idx]]
        self.labels = noisy_labels

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

        # Re-encode labels to be zero-indexed if not all values from 0 to max are present
        unique_labels = np.unique(self.labels)
        if not np.array_equal(unique_labels, np.arange(unique_labels.min(), unique_labels.max() + 1)):
            label_encoder = preprocessing.LabelEncoder()
            self.labels = torch.tensor(label_encoder.fit_transform(self.labels))
            self.class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

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
        self.labels = torch.tensor(label_encoder.fit_transform(self.labels))
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
