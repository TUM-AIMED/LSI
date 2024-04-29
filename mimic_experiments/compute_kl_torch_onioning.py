from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from utils.kl_div import _computeKL
import torch
from torch.utils.data import TensorDataset
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

gc.collect()
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(512, n_classes)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x

def get_mean_and_prec(data, labels, tinymodel):
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels).to(torch.long).to(DEVICE)
    data = np.asarray(data)
    data = torch.from_numpy(data).to(torch.float32).to(DEVICE)
    
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    # print(DEVICE)
    la = Laplace(tinymodel.features.to(DEVICE), 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)

    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=2500, help="Value for lerining_rate (optional)")
    parser.add_argument("--rem_small", type=bool, default=True, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem_per_step", type=int, default=1000, help="Value for lerining_rate (optional)")

    parser.add_argument("--corrupt", type=float, default=0.0)
    args = parser.parse_args()

    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Primacompressed":
        n_classes = 3
        args.subset = 4646
        args.n_rem = 4646
        args.range1 = 0
        args.range2 = 4646

    rem_small = args.rem_small
    n_rem_per_step = args.n_rem_per_step

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2"
    if rem_small:
        args.name_ext = "rem_small"
    else:
        args.name_ext = "rem_large"

    file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_onioning_" + "_dataset_" + str(args.dataset) + "_n_rem_per_step_" + str(args.n_rem_per_step) + "_" + str(args.name_ext)

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    keep_indices = [*range(args.subset)]
    remaining_indices = keep_indices
    data_set.reduce_to_active(keep_indices)
    X_train_init = data_set.data.to(DEVICE)
    y_train_init = data_set.labels.to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)
    seed = 1

    N_SEEDS = args.n_seeds
    epochs = args.epochs
    i = 0
    kl = []
    idx = []
    pred = []
    square_diff = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    onioning_ordering = []
    onioning_kl = []
    # selu_train_model = jax.jit(train_model)

    while len(remaining_indices) != 0:
        X_train = deepcopy(X_train_init)[remaining_indices]
        y_train = deepcopy(y_train_init)[remaining_indices]
        X_train = X_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        torch.manual_seed(seed + 5)
        model = TinyModel(n_classes)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.004,
        weight_decay=0.01, momentum=0.9, nesterov=True)
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model)
        # print(f"{time.time() - start_time}")
        kl_subset = []
        idx_subset = []
        pred_seed = []
        for i, idx in tqdm(enumerate(remaining_indices)):
            X_train_rem = torch.cat([X_train[0:i], X_train[i+1:]])
            y_train_rem = torch.cat([y_train[0:i], y_train[i+1:]])
            X_train_rem = X_train_rem.to(DEVICE)
            y_train_rem = y_train_rem.to(DEVICE)
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.004,
            weight_decay=0.01, momentum=0.9, nesterov=True)
            for i in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_rem)
                loss = criterion(output, y_train_rem)
                loss.backward()
                optimizer.step()
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model)
            kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
            kl_subset.append(kl1)
            square_diff.append(square_diff1)
            idx_subset.append(idx)
            # pred_seed.append(correct)
        combined = list(zip(kl_subset, idx_subset))
        if rem_small == True:
            combined.sort(key=lambda x: x[0])
        else: 
            combined.sort(key=lambda x: -x[0])

        kl_subset = combined[0]
        idx_subset = combined[1]
        if len(kl_subset) > n_rem_per_step:
            onioning_ordering.append(idx_subset[0:n_rem_per_step])
            onioning_kl.append(kl_subset[0:n_rem_per_step])
            remaining_indices = [x for x in remaining_indices if x not in idx_subset[0:n_rem_per_step]]
        else:
            onioning_ordering.extend(idx_subset)
            onioning_kl.extend(kl_subset)
            remaining_indices = [x for x in remaining_indices if x not in idx_subset[0:n_rem_per_step]]

    
    result = {"idx_ordering": onioning_ordering,
              "onioning_kl": onioning_kl}
    
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')