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

def corrupt_label(y_train, corrupt):
    y_unique = torch.unique(y_train)
    idx_list = list(range(len(y_train)))
    corrupted_idx = random.sample(idx_list, int(corrupt * len(y_train)))
    for idx in corrupted_idx:
        org_class = y_train[idx]
        sampled_class = random.randint(0, len(y_unique) - 1)
        while sampled_class == org_class:
            sampled_class = random.randint(0, len(y_unique) - 1)
        y_train[idx] = sampled_class
    return corrupted_idx, y_train


def corrupt_data(args, X_train, y_train, noise_level, corrupt_data_label):
    if args.dataset == "Imdbcompressed":
        n_corrupt = int(len(X_train) * noise_level)
        lorem_data_set_class, lorem_data_path = get_dataset("Loremcompressed")
        lorem_dataset = lorem_data_set_class(lorem_data_path, train=True)
        X_corrupt = lorem_dataset.data[0:n_corrupt].to(DEVICE)
        y_corrupt = torch.cat((torch.ones(n_corrupt // 2, dtype=torch.int), torch.zeros(n_corrupt // 2, dtype=torch.int)))
        y_corrupt = y_corrupt[torch.randperm(n_corrupt)].to(DEVICE)
        X_train_new = torch.concat([X_train, X_corrupt])
        y_train_new = torch.concat([y_train, y_corrupt])
        corrupted_idx = [i for i in range(len(X_train), len(X_train) + len(X_train_new))]
        return corrupted_idx, X_train_new, y_train_new
    else:
        X_train_new = []
        corrupted_idx = []
        std = torch.std(X_train)
        for i, (X_train_indiv, y_train_indiv) in enumerate(zip(X_train, y_train)):
            if y_train_indiv == corrupt_data_label:
                X_train_indiv += std*noise_level*torch.randn(X_train_indiv.shape).to(DEVICE)
                corrupted_idx.append(i)
            X_train_new.append(X_train_indiv)
        X_train_new = torch.stack(X_train_new)
        X_train_new = X_train_new.to(DEVICE)



    return corrupted_idx, X_train_new


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
    parser.add_argument("--model", type=str, default="TinyModel", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=500, help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=float, default=0.0)
    parser.add_argument("--corrupt_data", type=float, default=0.0)
    parser.add_argument("--corrupt_data_label", type=int, default=0)

    args = parser.parse_args()

    mom = 0.9 # args.mom # 0.9
    wd = 5e-4 # 0.01

    if args.dataset == "Imagenettecompressed":
        args.subset = 9469
        args.range1 = 0
        args.range2 = 9469
    if args.dataset == "Imagewoofcompressed":
        args.subset = 9025
        args.range1 = 0
        args.range2 = 9025
    if args.dataset == "Primacompressed":
        args.subset = 4646
        args.n_rem = 4646
        args.range1 = 0
        args.range2 = 4646
    if args.dataset == "Imdbcompressed":
        args.subset = 25000
        args.n_rem = 25000
        args.range1 = 0
        args.range2 = 25000


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_vs_diag_vs_kfac2"
    if args.name == None and args.range1 == 0 and args.range2 == 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset)  + "_model_" + str(args.model) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt)  + "_corrupt_data_" + str(args.corrupt_data) + "_" + str(args.corrupt_data_label) + "_" + str(args.name_ext)
    if args.name == None and args.range1 != 0 or args.range2 != 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset)  + "_model_" + str(args.model) + "_subset_" + str(args.subset) + "_range_" + str(args.range1) + "_" + str(args.range2) + "_corrupt_" + str(args.corrupt) + "_corrupt_data_" + str(args.corrupt_data) + "_" + str(args.corrupt_data_label) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    if args.dataset == "cifar10":
        data_set = data_set_class(data_path, train=True, ret4=False)
        data_set_test = data_set_class(data_path, train=False, ret4=False) 
    else:
        data_set = data_set_class(data_path, train=True)
        data_set_test = data_set_class(data_path, train=False) 
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    if args.dataset == "Imagenet":
        X_train = torch.tensor(data_set.data).to(DEVICE)
        y_train = torch.tensor(data_set.labels).to(DEVICE)
        X_test = torch.tensor(data_set_test.data).to(DEVICE)
        y_test = torch.tensor(data_set_test.labels).to(DEVICE)
    else:
        X_train = data_set.data.to(DEVICE)
        y_train = data_set.labels.to(DEVICE)
        X_test = data_set_test.data.to(DEVICE)
        y_test = data_set_test.labels.to(DEVICE)
    n_classes = len(torch.unique(y_test))

    corrupted_idx = None
    if args.corrupt > 0:
        corrupted_idx, y_train = corrupt_label(y_train, args.corrupt)
    if args.corrupt_data > 0:
        if args.dataset == "Imdbcompressed":
            corrupted_idx, X_train, y_train = corrupt_data(args, X_train, y_train, args.corrupt_data, args.corrupt_data_label)
            args.subset = len(X_train)
            args.range2 = len(X_train)
            args.n_rem = len(X_train)
        else:
            corrupted_idx, X_train = corrupt_data(args, X_train, y_train, args.corrupt_data, args.corrupt_data_label)



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
        mean1k, prec1k = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="kfac")
    
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
            mean2f, prec2f = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="full")
            print("Comp from full")
            kl1f, square_diff1f  = _computeKL_from_full(mean1f, mean2f, prec1f, prec2f)
            mean2d, prec2d = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="diag")
            print("Comp from diag")
            kl1d, square_diff1d = _computeKL(mean1d, mean2d, prec1d, prec2d)
            print("Comp from kfac")
            mean2k, prec2k = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="kfac")
            kl1k, square_diff1k = _computeKL_from_kron(mean1k, mean2k, prec1k, prec2k)
         
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
              "corrupted_idx": corrupted_idx}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')