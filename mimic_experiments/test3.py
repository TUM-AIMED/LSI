import torch
import torch.nn as nn
import torch.optim as optim
from compute_kl_jax import get_mean_and_prec
from utils.kl_div import _computeKL, _computeblockKL, _computeKL_from_full
from laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the dimensions of input, hidden, and output layers
input_size = 2
hidden_size = 2
output_size = 1

# Create an instance of the neural network
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example usage of the model with dummy data
# Dummy input and output tensors
data_samples = torch.tensor([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8],
    [0.2, 0.1],
    [0.6, 0.5],
    [0.8, 0.7],
    [0.9, 0.3],
    [0.4, 0.9],
    [0.2, 0.7],
    [0.1, 0.5],
    [0.6, 0.3],
    [0.3, 0.8],
    [0.5, 0.2],
    [0.8, 0.1],
    [0.4, 0.6],
    [0.7, 0.4],
    [0.9, 0.2],
    [0.2, 0.4],
    [0.1, 0.9]
])

# Calculate labels as the sum of the two dimensions
labels = torch.sum(data_samples, dim=1).unsqueeze(1)


# Training loop (You should provide your actual dataset and implement the training loop)
for epoch in range(1000):
    # Forward pass
    outputs = model(data_samples)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

la = Laplace(model, 'classification',
            subset_of_weights='all',
            hessian_structure='full')
# Create a TensorDataset with both data and labels
dataset = TensorDataset(data_samples, labels)

# Define batch size
batch_size = len(data_samples)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
la.fit(dataloader)

mean1 = la.mean.cpu().numpy()
post_prec1 = la.posterior_precision.cpu().numpy()

# data_samples = torch.cat((data_samples, torch.tensor([[100, 100]])))
# labels = torch.cat((labels, torch.tensor([[0.1]])))

data_samples = torch.cat((data_samples, torch.tensor([[0.3, 0.3]])))
labels = torch.cat((labels, torch.tensor([[0.6]])))


for epoch in range(1000):
    # Forward pass
    outputs = model(data_samples)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

la = Laplace(model, 'classification',
            subset_of_weights='all',
            hessian_structure='full')
# Create a TensorDataset with both data and labels

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
la.fit(dataloader)

mean2 = la.mean.cpu().numpy()
post_prec2 = la.posterior_precision.cpu().numpy()

kl1, square_diff1 = _computeKL_from_full(mean1, mean2, post_prec1, post_prec2)
print("")
