import torch
import torch.nn as nn
import torch.optim as optim
import os
from compute_kl_jax import get_mean_and_prec
from LSI.experiments.utils_kl import _computeKL, _computeblockKL, _computeKL_from_full
from laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from Datasets.dataset_helper import get_dataset

torch.manual_seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.use_deterministic_algorithms(True)
num_epochs = 1000
n_remove = 10
hessian_structure = "full"

# Define transformations to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize data to have mean 0 and standard deviation 1
])

# Download and load CIFAR-10 training dataset
train_dataset, train_path =   data_set_class, data_path = get_dataset("Prima")
train_dataset = data_set_class(data_path, train=True)
# data_set_test = data_set_class(data_path, train=False) 
labels = train_dataset.labels
data_samples = train_dataset.data



# Create an instance of the neural network
model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
model_unchanged = deepcopy(model)
model = model # .cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)

dataset = TensorDataset(data_samples, labels)
batch_size = len(data_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data_samples)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

la = Laplace(model, 'classification',
            subset_of_weights='all',
            hessian_structure=hessian_structure)
# Create a TensorDataset with both data and labels

la.fit(dataloader)
mean1 = la.mean.cpu().numpy()
post_prec1 = la.posterior_precision.cpu().numpy()

kl = []
square_diff = []
for i in tqdm(range(n_remove)):
    model2 = deepcopy(model_unchanged)
    model2 = model2.cuda()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.1, weight_decay=1e-5)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model2(torch.cat((data_samples[:i], data_samples[i+1:])))
        loss = criterion(outputs, torch.cat((labels[:i], labels[i+1:])))
        
        # Backward pass and optimization
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        
        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    la = Laplace(model2, 'classification',
                subset_of_weights='all',
                hessian_structure=hessian_structure)
    # Create a TensorDataset with both data and labels

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    la.fit(dataloader)

    mean2 = la.mean.cpu().numpy()
    post_prec2 = la.posterior_precision.cpu().numpy()
    if hessian_structure == "diag":
        kl1, square_diff1 = _computeKL(mean1, mean2, post_prec1, post_prec2)
    elif hessian_structure == "full":
        kl1, square_diff1 = _computeKL_from_full(mean1, mean2, post_prec1, post_prec2)
    else:
        raise Exception("Not implemented")
    kl.append(kl1)
    square_diff.append(square_diff1)

correlation_coefficient, p_value = pearsonr(kl, square_diff)
print("Pearson correlation coefficient:", correlation_coefficient)
print("P-value:", p_value)

    # Create scatter plot
plt.scatter(kl, square_diff)

# Add labels and title
plt.xlabel('KL')
plt.ylabel('Square_diff')

# Save the plot as a PNG file
plt.savefig('./scatter_plot.png')
with open('./full.pkl', 'wb') as file:
    pickle.dump([kl, square_diff], file)
print("")
