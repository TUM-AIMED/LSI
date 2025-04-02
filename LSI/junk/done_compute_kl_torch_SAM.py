from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from LSI.experiments.utils_kl import _computeKL, _computeKL_from_full
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
from utils.sam import SAM


gc.collect()
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class TinyModel(torch.nn.Module):
    def __init__(self, n_classes, in_features=512):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, n_classes)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x

def get_mean_and_prec(data, labels, model, mode = "diag"):
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
    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec

def corrupt_label(y_train, corrupt):
    y_unique = torch.unique(y_train)
    idx_list = list(range(len(y_train)))
    corrupted_idx = random.sample(idx_list, int(corrupt * len(y_train)))
    # corrupted_idx = list(range(500))
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
        # y_corrupt = lorem_dataset.labels[torch.randperm(n_corrupt)].to(DEVICE)
        # test = y_corrupt.unique(return_counts=True)
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
    parser.add_argument("--name_ext", type=str, default="SAM")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)") # 150 for Resnet, 700 for CNN, 600 for MLP
    parser.add_argument("--lr", type=float, default=0.004, help="Value for lerining_rate (optional)") # General 0.004 0.01 for Resnet, 0.005 for CNN and MLP
    parser.add_argument("--mom", type=float, default=0.9, help="Value for lerining_rate (optional)") # 0.9

    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="Tinymodel", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--full", action="store_true", default=False, help="A boolean argument")
    parser.add_argument("--corrupt", type=float, default=0.0)
    parser.add_argument("--corrupt_data", type=float, default=0.0)
    parser.add_argument("--corrupt_data_label", type=int, default=0)

    args = parser.parse_args()

    full = args.full
    print(f"Full? {full}")
    mom = 0.9 # args.mom # 0.9
    wd = 5e-4 # 0.01

    if args.dataset == "cifar10":
        n_classes = 4
    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Imagenettecompressed":
        n_classes = 10
        args.subset = 9469
        args.range1 = 0
        args.range2 = 9469
    if args.dataset == "Imagewoofcompressed":
        n_classes = 10
        args.subset = 9025
        args.range1 = 0
        args.range2 = 9025
    if args.dataset == "Primacompressed":
        n_classes = 3
        args.subset = 4646
        args.n_rem = 4646
        args.range1 = 0
        args.range2 = 4646
    if args.dataset == "Imdbcompressed":
        n_classes = 2
        args.subset = 25000
        args.n_rem = 25000
        args.range1 = 0
        args.range2 = 25000


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_SAM"
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


    if args.model == "Tinymodel":
        model_class = TinyModel
    else:
        model_class = get_model(args.model)


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
        if args.dataset == "Imdbcompressed":
            model = model_class(n_classes, in_features=768)
        else:
            model = model_class(n_classes)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
        model.parameters(),
        base_optimizer=base_optimizer,
        lr=args.lr,
        momentum=mom)
        for i in tqdm(range(epochs)):
            # optimizer.zero_grad()
            # output = model(X_train)
            # pred = torch.argmax(output, dim=1)
            # accuracy = torch.sum(pred == y_train)/y_train.shape[0]
            # print(accuracy)
            # loss = criterion(output, y_train)
            # print(loss)
            # loss.backward(retain_graph=True)
            # optimizer.first_step(zero_grad=True)
            # loss = criterion(output, y_train).backward()
            # optimizer.second_step(zero_grad=True)

            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward(retain_graph=True)
            optimizer.first_step(zero_grad=True)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            pred = torch.argmax(output, dim=1)
            accuracy = torch.sum(pred == y_train)/y_train.shape[0]
            print(accuracy)
            print(loss)
        if full:
            mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="full")
        else:
            mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model)
        # print(f"{time.time() - start_time}")
        kl_seed = []
        idx_seed = []
        pred_seed = []
        for i in tqdm(range(N_REMOVE1, N_REMOVE2)):
            X_train_rem = torch.cat([X_train[0:i], X_train[i+1:]])
            y_train_rem = torch.cat([y_train[0:i], y_train[i+1:]])
            X_train_rem = X_train_rem.to(DEVICE)
            y_train_rem = y_train_rem.to(DEVICE)
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            base_optimizer2 = torch.optim.SGD
            optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer2,
            lr=args.lr,
            momentum=mom)
            for j in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_rem)
                loss = criterion(output, y_train_rem)
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)
                loss = criterion(output, y_train_rem)
                loss.backward()
                optimizer.second_step(zero_grad=True
                                      )
            if full:
                mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="full")
                kl1, square_diff1  = _computeKL_from_full(mean1, mean2, prec1, prec2)
            else:
                mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model)
                kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
            print(f"KL divergence {kl1}")
            kl_seed.append(kl1)
            square_diff.append(square_diff1)
            idx_seed.append(i)
        kl.append(kl_seed)
        idx.append(idx_seed)

    result = {"idx": idx,
              "kl": kl,
              "pred":pred, 
              "square_diff":square_diff,
              "corrupted_idx": corrupted_idx}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')