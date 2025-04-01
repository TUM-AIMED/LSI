print("wtf is going on")
# import torch.nn as nn
# from Datasets.dataset_cifar10_regression_task import CIFAR10
print("wtf is going on2")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # Assuming input image size is reduced to 8x8 after 3 conv layers
        self.fc2 = nn.Linear(256, 1)  # Output size is 1 for regression task

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Reshape for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == "__main__":


    print("here8")
    data_set = CIFAR10("/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR10/cifar-10-batches-py")
    data_set_test = CIFAR10("/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR10/cifar-10-batches-py", train=False)
    batch_size = 64
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    
    data_sample = data_set[0][0]
    label_sample = data_set[0][1]
    input_size = 3072
    hidden_size1 = 512
    hidden_size2 = 64
    output_size = 1
    # model = MLP(input_size, hidden_size1, hidden_size2, output_size)
    print("here6")
    model = CNN()
    print("here7")
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs.to(DEVICE)
            targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(data_set)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")











    

