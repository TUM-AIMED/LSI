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
from joblib import Parallel, delayed, cpu_count
import time

print(cpu_count())

# gc.collect()
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda")


# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

print(DEVICE)


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

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=len(labels),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # print(DEVICE)
    la = Laplace(tinymodel.features, 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)

    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec

def split_into_chunks(lst, n):
    # Calculate the size of each chunk
    chunk_size = len(lst) // n
    remainder = len(lst) % n

    chunks = []
    start = 0
    for i in range(n):
        # Adjust the chunk size if there's a remainder
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size

        chunks.append(lst[start:end])
        start = end

    return chunks

def train_model(idxs, real_idxs, mean1, prec1, X_train, y_train):
    kl_par = []
    square_diff_par = []
    ret_real_idxs_par = []
    for idx, real_idx in zip(idxs, real_idxs):
        X_train_rem = torch.cat([X_train[0:idx], X_train[idx+1:]])
        y_train_rem = torch.cat([y_train[0:idx], y_train[idx+1:]])
        model = deepcopy(backup_model)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.004,
            weight_decay=0.01, momentum=0.9, nesterov=True)
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X_train_rem)
            loss = criterion(output, y_train_rem)
            loss.backward()
            optimizer.step()
        mean2, prec2 = get_mean_and_prec(X_train_rem, y_train_rem, model)
        kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
        kl_par.append(kl1)
        square_diff_par.append(square_diff1)
        ret_real_idxs_par.append(real_idx)
    return kl_par, square_diff_par, ret_real_idxs_par


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=700, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--rem_small", type=bool, default=True, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem_per_step", type=int, default=5000, help="Value for lerining_rate (optional)")

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

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_onioning"
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
        n_jobs = 60
        indices_in_data = list(range(len(remaining_indices)))
        chunks = split_into_chunks(remaining_indices, n_jobs * 8)
        i_chunks = split_into_chunks(indices_in_data, n_jobs * 8)
        start_time = time.time()
        results = Parallel(n_jobs=-2, batch_size=1, verbose=20)(
            delayed(train_model)(i_s, chunk, mean1, prec1, X_train, y_train) for i_s, chunk in tqdm(zip(i_chunks, chunks))
        )
        print(f"took {time.time() - start_time}")
        for kl1, square_diff1, idx in results:
            kl_subset.extend(kl1)
            square_diff.extend(square_diff1)
            idx_subset.extend(idx)
        combined = list(zip(kl_subset, idx_subset))
        if rem_small == True:
            combined.sort(key=lambda x: x[0])
        else: 
            combined.sort(key=lambda x: -x[0])

        kl_subset, idx_subset = zip(*combined)
        kl_subset = list(kl_subset)
        idx_subset = list(idx_subset)
        if len(kl_subset) > n_rem_per_step:
            onioning_ordering.extend(idx_subset[0:n_rem_per_step])
            onioning_kl.extend(kl_subset[0:n_rem_per_step])
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