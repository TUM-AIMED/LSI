from torch.utils.data import Dataset
import numpy as np
import torch

class NoisyDataset(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None, noisy_idx=[]):
        super(NoisyDataset, self).__init__(data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None)
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



class BatchDatasetWrapper:
    def __init__(self, dataset, num_batches):
        """
        Wraps a dataset and splits it into batches.

        Args:
            dataset (CompressedDataset): The dataset to wrap.
            num_batches (int): Number of batches to split the dataset into.
        """
        self.dataset = dataset
        self.num_batches = num_batches
        self.data = []
        self.label = []

        self._split_into_batches()

    def _split_into_batches(self):
        """
        Splits the dataset into batches of equal size.
        """
        data = self.dataset.data
        labels = self.dataset.labels
        batch_size = len(data) // self.num_batches

        self.data = [data[i * batch_size:(i + 1) * batch_size] for i in range(self.num_batches)]
        self.label = [labels[i * batch_size:(i + 1) * batch_size] for i in range(self.num_batches)]

        # Handle any remaining data
        if len(data) % self.num_batches != 0:
            self.data[-1] = torch.cat((self.data[-1], data[self.num_batches * batch_size:]))
            self.label[-1] = torch.cat((self.label[-1], labels[self.num_batches * batch_size:]))

    def remove_index(self, index):
        """
        Removes an index from the correct batch, leaving other batches unchanged.

        Args:
            index (int): The global index to remove.
        """
        # Find the batch and local index
        cumulative_size = 0
        for batch_idx, batch in enumerate(self.data):
            if cumulative_size + len(batch) > index:
                local_index = index - cumulative_size
                self.data[batch_idx] = torch.cat((batch[:local_index], batch[local_index + 1:]))
                self.label[batch_idx] = torch.cat((self.label[batch_idx][:local_index], self.label[batch_idx][local_index + 1:]))
                break
            cumulative_size += len(batch)