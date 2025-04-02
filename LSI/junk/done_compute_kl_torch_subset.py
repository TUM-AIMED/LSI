from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from LSI.experiments.utils_kl import _computeKL, _computeKL_from_full, _computeKL_from_kron, kl_divergence_kron_ala_gpt
import torch
from torch.utils.data import TensorDataset
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from LSI.models.models import get_model
import random 


gc.collect()
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

# Set deterministic and disable benchmarking
def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mean_and_prec(data, labels, model, mode):
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
    if mode == "diag":
        la = Laplace(model.features.to(DEVICE), 'classification',
                    subset_of_weights='all',
                    hessian_structure='diag')
        la.fit(train_loader)
    elif mode == "full":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights='all',
            hessian_structure='full')
        la.fit(train_loader)
    elif mode == "kfac":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights='all',
            hessian_structure='kron')
        la.fit(train_loader)
    if mode == "kfac":
        mean = la.mean.cpu().numpy()
        post_prec = la.posterior_precision
        post_prec.eigenvalues = [[tensor.cpu() for tensor in ev] for ev in post_prec.eigenvalues]
        post_prec.eigenvectors = [[tensor.cpu() for tensor in ev] for ev in post_prec.eigenvectors]
    else:
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
    parser.add_argument("--n_rem", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--batch_num", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)") # 150 for Resnet, 700 for CNN, 600 for MLP
    parser.add_argument("--lr", type=float, default=0.004, help="Value for lerining_rate (optional)") # General 0.004 0.01 for Resnet, 0.005 for CNN and MLP
    parser.add_argument("--mom", type=float, default=0.9, help="Value for lerining_rate (optional)") # 0.9

    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="MiddleModel", help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=1000, help="Value for lerining_rate (optional)")


    args = parser.parse_args()

    mom = 0.9 # args.mom # 0.9
    wd = 5e-4 # 0.01

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_subset"
    file_name = "subset_w_full"

    data_set_class, data_path = get_dataset(args.dataset)


    data_set = data_set_class(data_path, train=True)
    data_set.reduce_to_active_class([0, 1, 2])
    data_set.reduce_to_active([i for i in range(1000)])
    data_set_test = data_set_class(data_path, train=False) 
    data_set_test.reduce_to_active_class([0, 1, 2])

    active_idx = data_set.active_indices

    X_train = data_set.data.to(DEVICE)
    y_train = torch.tensor(data_set.labels).to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = torch.tensor(data_set_test.labels).to(DEVICE)


    n_classes = len(torch.unique(y_test))

    model_class = get_model(args.model)


    N_REMOVE1 = 0
    N_REMOVE2 = args.n_rem

    if args.range1 != 0 or args.range2 != 0:
        N_REMOVE1 = args.range1
        N_REMOVE2 = args.range2
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    i = 0
    
    klf = []
    kld = []
    klk = []
    idx = []
    pred = []
    square_diff = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for seed in range(N_SEEDS):
        set_seed(seed + 5)
        set_deterministic()
        if args.dataset == "Imdbcompressed":
            model = model_class(n_classes, in_features=768)
        else:
            model = model_class(n_classes)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=wd, momentum=mom, nesterov=True)
        for i in tqdm(range(epochs)):
            optimizer.zero_grad()
            output = model(X_train)
            pred = torch.argmax(output, dim=1)
            accuracy = torch.sum(pred == y_train)/y_train.shape[0]
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        mean1f, prec1f = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="full")
        mean1d, prec1d = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="diag")
        # mean1k, prec1k = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="kfac")
    
        klf_seed = []
        kld_seed = []
        klk_seed = []
        idx_seed = []
        pred_seed = []
        for i in tqdm(range(N_REMOVE1, N_REMOVE2)):
            X_train_rem = torch.cat([X_train[0:i], X_train[i+1:]])
            y_train_rem = torch.cat([y_train[0:i], y_train[i+1:]])
            X_train_rem = X_train_rem.to(DEVICE)
            y_train_rem = y_train_rem.to(DEVICE)
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=wd, momentum=mom, nesterov=True)
            for j in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_rem)
                loss = criterion(output, y_train_rem)
                loss.backward()
                optimizer.step()
            kl1f = 0
            kl1k = 0
            mean2f, prec2f = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="full")
            print("Comp from full")
            kl1f, square_diff1f  = _computeKL_from_full(mean1f, mean2f, prec1f, prec2f)
            mean2d, prec2d = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="diag")
            print("Comp from diag")
            kl1d, square_diff1d = _computeKL(mean1d, mean2d, prec1d, prec2d)
            # print("Comp from kfac")
            # mean2k, prec2k = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="kfac")
            # kl1k, square_diff1k = _computeKL_from_kron(mean1k, mean2k, prec1k, prec2k)
         
            print(f"KL divergence {kl1f}")
            print(f"KL divergence {kl1d}")
            print(f"KL divergence {kl1k}")
            klf_seed.append(kl1f)
            kld_seed.append(kl1d)
            klk_seed.append(kl1k)
            idx_seed.append(i)
        klf.append(klf_seed)
        kld.append(kld_seed)
        klk.append(klk_seed)
        idx.append(idx_seed)

    
    result = {"idx": idx,
              "klf": klf,
              "klk": klk,
              "kld": kld,
              "pred":pred, 
              "square_diff":square_diff,
              "active_idx": active_idx}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')