import torch
from torch.utils.data import Dataset
import numpy as np
import warnings
import pandas as pd
import os


# Define a custom PyTorch dataset class
class DiabetesDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, classes=None, portions=None):
        df_raw = pd.read_csv(os.path.join(data_path, 'diabetic_data.csv'))
        df_raw = df_raw[df_raw["readmitted"] != "NO"]
        df_raw.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], inplace=True)
        first_row_number = df_raw.index[0]
        # Extract features and labels
        for column_name in df_raw.columns:
            unique_categories = df_raw[column_name].unique().tolist()
            unique_count = df_raw[column_name].value_counts().tolist()
            if len(unique_categories) == 1:
                df_raw.drop(columns=column_name, inplace=True)
                continue
            # print(unique_categories)
            # print([element / sum(unique_count) for element in unique_count])
            if type(df_raw[column_name][first_row_number]) == str or column_name == "max_glu_serum" or column_name == "A1Cresult":
                assignment =  [*range(len(unique_categories))]
                df_raw[column_name].replace(unique_categories,
                            assignment, inplace=True)


        # Convert features and labels to PyTorch tensors
        features = df_raw.iloc[:, :-1].values  # Select all columns except the last one
        labels = df_raw.iloc[:, -1].values    # Select the last column
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.unbalanced_idx = [*range(self.features.shape[1])]

        seed = 42
        torch.manual_seed(seed)
        train_size = int(0.8 * len(self.features))
        test_size = len(self.features) - train_size

        min_vals, _ = torch.min(self.features, dim=0)
        max_vals, _ = torch.max(self.features, dim=0)
        test2 = (max_vals - min_vals)

        # Apply min-max normalization to the tensor
        self.features = (self.features - min_vals) / (max_vals - min_vals)

        train_set, test_set = torch.utils.data.random_split(self.features, [train_size, test_size])
        if train:
            self.features = train_set
            self.unbalanced_idx = np.array(self.features.indices)
            self.labels = self.labels[self.unbalanced_idx]
            self.class_assignment_list = self.labels
        else:
            self.features = test_set
            self.unbalanced_idx = np.array(self.features.indices)
            self.labels = self.labels[self.unbalanced_idx]
            self.class_assignment_list = self.labels
        # Find the minimum and maximum values along each feature dimension


    def reduce_to_active(self, indexes):
        self.unbalanced_idx = self.unbalanced_idx[indexes]
        self.features = self.features[indexes]
        self.labels = self.labels[indexes]
        self.class_assignment_list = self.labels



    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y, index
    