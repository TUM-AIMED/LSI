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
    parser.add_argument("--n_rem", type=int, default=10000, help="Value for lerining_rate (optional)")
    parser.add_argument("--batch_num", type=int, default=3, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=400, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=0.04, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=100, help="Value for lerining_rate (optional)")
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


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2"
    if args.name == None and args.range1 == 0 and args.range2 == 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt) + "_batched_" + str(args.batch_num) + "_" + str(args.name_ext)
    if args.name == None and args.range1 != 0 or args.range2 != 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_range_" + str(args.range1) + "_" + str(args.range2) + "_corrupt_" + str(args.corrupt) + "_batched_" + str(args.batch_num) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    X_train = data_set.data
    y_train = data_set.labels
    batched_X = list(X_train.chunk(args.batch_num))
    batched_X = [batch.to(DEVICE) for batch in batched_X]
    batched_y = list(y_train.chunk(args.batch_num))
    batched_y = [batch.to(DEVICE) for batch in batched_y]

    idx_batch_assignment = torch.tensor(keep_indices).chunk(args.batch_num)


    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)

    N_REMOVE1 = 0
    N_REMOVE2 = args.n_rem

    if args.range1 != 0 or args.range2 != 0:
        N_REMOVE1 = args.range1
        N_REMOVE2 = args.range2
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    i = 0
    kl = []
    idx = []
    pred = []
    square_diff = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # selu_train_model = jax.jit(train_model)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed + 5)
        model = TinyModel(n_classes)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01, momentum=0.9, nesterov=True)
        for i in range(epochs):
            for batch_X, batch_y in zip(batched_X, batched_y):
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        mean1, prec1 = get_mean_and_prec( torch.cat(batched_X).cpu(),  torch.cat(batched_y).cpu(), model)
        # print(f"{time.time() - start_time}")
        kl_seed = []
        idx_seed = []
        pred_seed = []
        for i in tqdm(range(N_REMOVE1, N_REMOVE2)):
            remove_index_in_batched = [torch.where(batch == i)[0] for batch in idx_batch_assignment]
            rem_batch = [i for i, val in enumerate(remove_index_in_batched) if val.nbytes != 0][0]
            index_in_batch = remove_index_in_batched[rem_batch].item()
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            batched_X_rem = deepcopy(batched_X)
            batched_y_rem = deepcopy(batched_y)
            batched_X_rem[rem_batch] = torch.cat([batched_X_rem[rem_batch][0:index_in_batch], batched_X_rem[rem_batch][index_in_batch+1:]]) 
            batched_y_rem[rem_batch] = torch.cat([batched_y_rem[rem_batch][0:index_in_batch], batched_y_rem[rem_batch][index_in_batch+1:]]) 
            optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01, momentum=0.9, nesterov=True)
            for i in range(epochs):
                for batch_X, batch_y in zip(batched_X_rem, batched_y_rem):
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            mean2, prec2 = get_mean_and_prec(torch.cat(batched_X_rem).cpu(), torch.cat(batched_y_rem).cpu(), model)
            kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
            # kl1, square_diff1 = _computeKL_from_full(mean1, mean2, prec1, prec2)
            print(f"KL divergence {kl1}")
            # prediction = model.predict(X_train)
            # pred_class = jnp.argmax(prediction, axis=1)
            # correct = pred_class == y_train
            kl_seed.append(kl1)
            square_diff.append(square_diff1)
            idx_seed.append(i)
            # pred_seed.append(correct)
        kl.append(kl_seed)
        idx.append(idx_seed)
        # pred.append(correct)
        # print(f"loop took {time.time() - start_time}")
        
    result = {"idx": idx,
              "kl": kl,
              "pred":pred, 
              "square_diff":square_diff,
              "idx_batch_assignmet":idx_batch_assignment}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')