import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class GaussianDataset(Dataset):
    def __init__(self, file_path=None, num_samples=100, seed=42):
        if file_path is not None:
            self.data = pd.read_csv(file_path)
            self.labels = torch.tensor(self.data.iloc[:, -1], dtype=torch.long)
        else:
            np.random.seed(seed)
            class1_mean = [1, 1]
            class1_cov = [[1, 0], [0, 1]]
            class2_mean = [-1, -1]
            class2_cov = [[1, 0], [0, 1]]

            class1_data = np.random.multivariate_normal(class1_mean, class1_cov, num_samples)
            class2_data = np.random.multivariate_normal(class2_mean, class2_cov, num_samples)

            class1_labels = np.zeros(num_samples)
            class2_labels = np.ones(num_samples)

            data = np.concatenate([class1_data, class2_data], axis=0)
            labels = np.concatenate([class1_labels, class2_labels])

            self.data = pd.DataFrame(data, columns=['feature1', 'feature2'])
            self.data['label'] = labels.astype(int)

            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return features, label

    def save_to_csv(self, file_path):
        self.data.to_csv(file_path, index=False)

# Example usage:
file_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/Datasets/gaussian_dataset.csv'
dataset = GaussianDataset(file_path=file_path)
# dataset.save_to_csv(file_path)
df = dataset.data  # Assuming you already have the dataset loaded

# Separate data points based on labels
class1 = df[df['label'] == 0]
class2 = df[df['label'] == 1]

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(class1['feature1'], class1['feature2'], color='blue', label='Class 0')
plt.scatter(class2['feature1'], class2['feature2'], color='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Feature 1 vs Feature 2')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('scatter_plot.png')
print("")