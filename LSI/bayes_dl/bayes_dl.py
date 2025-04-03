import sys
sys.path.append('/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments')
from bayesdll import vi, mc_dropout, sgld, la
from bayesdll.vi import Runner
from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from LSI.experiments.utils_kl import _computeKL
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import logging


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

def get_mean_and_prec(runner):
    a_std = list(runner.model._retrieve_s().parameters())
    a_mean = list(runner.model.m.parameters())
    l1_std = [params.view(-1).tolist() for params in a_std]
    l1_mean = [params.view(-1).tolist() for params in a_mean]
    l_std = []
    l_mean = []
    [l_std.extend(sublist) for sublist in l1_std]
    [l_mean.extend(sublist) for sublist in l1_mean]
    n_std = np.array(l_std)
    n_mean = np.array(l_mean)
    n_var = n_std**2
    n_prec = 1/n_var
    return n_mean, n_prec

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem", type=int, default=10000, help="Value for lerining_rate (optional)")
    parser.add_argument("--batch_num", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=100, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default= 0.001, help="Value for lerining_rate (optional)")
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
    args.pretrained = None
    args.device = DEVICE
    args.lr_head = args.lr
    args.hparams = {}
    args.hparams["prior_sig"]=1.0
    args.hparams["kld"]=1e-3
    args.hparams["bias"]="informative"
    args.hparams["nst"]=5
    args.ece_num_bins = 15
    args.epochs = 1000
    args.momentum = 0.0
    args.seed = 42
    args.test_eval_freq = 25
    args.log_dir = 'results'
    args.ND = args.subset

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2"
    if args.name == None and args.range1 == 0 and args.range2 == 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt) + "_" + str(args.name_ext)
    if args.name == None and args.range1 != 0 or args.range2 != 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_range_" + str(args.range1) + "_" + str(args.range2) + "_corrupt_" + str(args.corrupt) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    logging.basicConfig(
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, 'logs.txt')), 
        logging.StreamHandler()
    ], 
    format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
    )
    logger = logging.getLogger()

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    X_train = data_set.data.to(DEVICE)
    y_train = data_set.labels.to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

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
        rand_state = torch.random.get_rng_state()
        model = TinyModel(n_classes)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        model.readout_name = 'classifier'
        runner = Runner(model, None, args, logger)
        runner.train(train_loader, test_loader, test_loader)
        n_mean, n_prec = get_mean_and_prec(runner)

        kl_seed = []
        idx_seed = []
        pred_seed = []
        for i in tqdm(range(N_REMOVE1, N_REMOVE2)):
            torch.manual_seed(seed + 5)
            torch.random.set_rng_state(rand_state)
            X_train_rem = torch.cat([X_train[0:i], X_train[i+1:]])
            y_train_rem = torch.cat([y_train[0:i], y_train[i+1:]])
            X_train_rem = X_train_rem.to(DEVICE)
            y_train_rem = y_train_rem.to(DEVICE)
            train_dataset_rem = TensorDataset(X_train_rem, y_train_rem)
            train_loader_rem = DataLoader(train_dataset_rem, batch_size=len(X_train_rem), shuffle=False)
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            model.readout_name = 'classifier'
            runner = Runner(model, None, args, logger)
            runner.train(train_loader_rem, test_loader, test_loader)
            n_mean2, n_prec2 = get_mean_and_prec(runner)

            kl1, square_diff1 = _computeKL(n_mean, n_mean2, n_prec, n_prec2)
            print(kl1)
            print(square_diff1)
            kl_seed.append(kl1)
            square_diff.append(square_diff1)
            idx_seed.append(i)
        kl.append(kl_seed)
        idx.append(idx_seed)

        
    plt.scatter(kl_seed, square_diff)

    # Add labels and title
    plt.xlabel('List 1')
    plt.ylabel('List 2')
    plt.title('Scatter Plot')

    # Save the plot as test.png
    plt.savefig('./test.png')
    
    result = {"idx": idx,
              "kl": kl,
              "pred":pred, 
              "square_diff":square_diff}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')