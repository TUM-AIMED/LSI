import torch
from laplace import Laplace
import numpy as np
from torch.utils.data import TensorDataset


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, n_classes)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.features = torch.nn.Sequential(self.linear1, self.relu1, self.linear2, self.relu2, self.linear3)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x


data = torch.rand((100, 512))
labels = torch.randint(0, 2, (100,))

train_loader = torch.utils.data.DataLoader(
    TensorDataset(data, labels),
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

model = TinyModel(2)

# print(DEVICE)
la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights='all',
            hessian_structure='diag')
la.fit(train_loader)

mean = la.mean.cpu().numpy()
post_prec = la.posterior_precision.cpu().numpy()
print("")
