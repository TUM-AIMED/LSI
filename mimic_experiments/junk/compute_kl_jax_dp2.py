import torch
import torch.nn as nn
from Datasets.dataset_cifar10_regression_task import CIFAR10
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file"""
    data_set = CIFAR10("/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR10/cifar-10-batches-py")
    data_set_test = CIFAR10("/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR10/cifar-10-batches-py", train=False)

    data = data_set.data
    data = data.reshape(data.shape[0], -1)
    targets = data_set.kl_data[0:40000]

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(data, targets)

    # Predict using the fitted model
    predictions = model.predict(data)

    # Calculate Mean Squared Error
    mse = mean_squared_error(targets, predictions)
    print("Mean Squared Error:", mse)

    # Print the coefficients
    print("Coefficients:", model.coef_)

    # Print the intercept
    print("Intercept:", model.intercept_)

    # batch_size = 512
    # dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # testloader = DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
    
    # data_sample = data_set[0][0]
    # label_sample = data_set[0][1]
    # input_size = 3072
    # hidden_size1 = 512
    # hidden_size2 = 64
    # output_size = 1
    # # model = MLP(input_size, hidden_size1, hidden_size2, output_size)
    # print("here6")
    # model = CNN()
    # print("here7")
    # model.to(DEVICE)
    # criterion = nn.MSELoss(reduction = 'sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # # Training loop
    # num_epochs = 1000
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     running_loss_test = 0.0
    #     for i, (inputs, targets) in enumerate(dataloader):
    #         inputs = inputs.to(DEVICE)
    #         targets = targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs.squeeze(), targets)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item() * inputs.size(0)
    #     epoch_loss = running_loss / len(data_set)
    #     for i, (inputs, targets) in enumerate(testloader):
    #         inputs = inputs.to(DEVICE)
    #         targets = targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs.squeeze(), targets)
    #         running_loss_test += loss.item() * inputs.size(0)
    #     epoch_loss_test = running_loss_test / len(data_set_test)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Loss test: {epoch_loss_test:.4f}")


    print("does this work now")