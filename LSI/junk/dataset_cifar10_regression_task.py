import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle


class CIFAR10(Dataset):
    def __init__(self, data_path, train=True, classes=None, portions=None, transform=None, shuffle=False, resize = None):
        self.smallest_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"
        with open(self.smallest_path, 'rb') as file:
            final_dict = pickle.load(file)
        self.kl_data = np.array(final_dict["kl"])
        self.kl_data = np.mean(self.kl_data, axis=0)
        # mean = np.mean(self.kl_data)
        # std_dev = np.std(self.kl_data)
        # self.kl_data = (self.kl_data - mean) / std_dev
        self.idx = final_dict["idx"][0]
        
        self.root_dir = data_path

        print("here1")

        data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        data = []
        labels = []
        print("here2")
        for file_name in data_files[0:100]:
            file = os.path.join(data_path, file_name)
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                data.extend(data_dict [b'data'])
                labels.extend(data_dict [b'labels'])
        
        print("here3")
          

        self.data = np.array(data)
        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.labels = np.array(labels)
        self.attributes = torch.empty((2,3), dtype=torch.int64)
        print("here4")

        if train:
            self.data = self.data[0:40000]
            self.labels = self.labels[0:40000]
            self.active_idx = list(range(0,40000))
        else:
            self.data = self.data[40000:50000]
            self.labels = self.labels[40000:50000]
            self.active_idx = list(range(40000,50000))
        
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx_in):
        idx = self.active_idx[idx_in]
        image = self.data[idx_in]
        label = self.labels[idx_in]
        image = image.transpose(1, 2, 0)
        transform = transforms.ToTensor()
        transform2 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        tensor_image = transform(image)
        tensor_image = transform2(tensor_image)
        index_in_kl = self.idx.index(idx)
        kl = self.kl_data[index_in_kl]
        kl = torch.tensor(kl.astype(np.float32))
       
        
        return tensor_image, kl

