from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from utils.kl_div import _computeKL, _computeKL_from_full, _computeKL_from_kron, kl_divergence_kron_ala_gpt
import torch
from torch.utils.data import TensorDataset
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from models.final_models_to_test import get_model
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

    
def get_mean_and_prec(data, labels, model, mode, subset = "all"):
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
                    subset_of_weights=subset,
                    hessian_structure='diag')
        la.fit(train_loader)
    elif mode == "full":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights=subset,
            hessian_structure='full')
        la.fit(train_loader)
    elif mode == "kfac":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights=subset,
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
    parser.add_argument("--batch_num", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=500, help="Value for lerining_rate (optional)") # 150 for Resnet, 700 for CNN, 600 for MLP
    parser.add_argument("--lr", type=float, default=0.01, help="Value for lerining_rate (optional)") # General 0.004 0.01 for Resnet, 0.005 for CNN and MLP
    parser.add_argument("--lr2", type=float, default=0.004, help="Value for lerining_rate (optional)") # General 0.004 0.01 for Resnet, 0.005 for CNN and MLP

    parser.add_argument("--mom", type=float, default=0.9, help="Value for lerining_rate (optional)") # 0.9
    parser.add_argument("--dataset", type=str, default="cifar10", help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="SmallestCNN", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=500, help="Value for lerining_rate (optional)")
    parser.add_argument("--laptype", type=str, default="diag", help="Value for lerining_rate (optional)")


    args = parser.parse_args()

    mom = 0.9 # args.mom # 0.9
    wd = 5e-4 # 0.01


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_all_layers_vs_last_layer"
    file_name = "full_vs_lastlayer_SmallCNN_all_three_500_convex"

    data_set_class, data_path = get_dataset(args.dataset)

    if args.dataset == "cifar10":
        data_set = data_set_class(data_path, train=True, ret4=False)
        data_set_test = data_set_class(data_path, train=False, ret4=False) 

    data_set.reduce_to_active_class([0, 1, 2])

    X_train = data_set.data.to(DEVICE)
    y_train = data_set.labels.to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)

    n_classes = len(torch.unique(y_test))

    if args.range1 != 0 or args.range2 != 0:
        N_REMOVE1 = args.range1
        N_REMOVE2 = args.range2
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    i = 0
    kl = []
    kl_ll = []
    idx = []
    pred = []
    square_diff = []
    kl_proxy = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    set_seed(5)
    set_deterministic()

    model_class = get_model(args.model)
    model = model_class(n_classes)
    backup_model = deepcopy(model)
    model = deepcopy(backup_model)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    weight_decay=wd, momentum=mom, nesterov=True)
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(X_train)
        # pred = torch.argmax(output, dim=1)
        # accuracy = torch.sum(pred == y_train)/y_train.shape[0]
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    if args.laptype == "full":
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="full")
    elif args.laptype =="diag":
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="diag")
    elif args.laptype =="kfac":
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="kfac")
    else:
        raise Exception("Not implemented")
    
    if args.laptype == "full":
        mean1_ll, prec1_ll = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="full", subset="last_layer")
    elif args.laptype =="diag":
        mean1_ll, prec1_ll = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="diag", subset="last_layer")
    elif args.laptype =="kfac":
        mean1_ll, prec1_ll = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, mode="kfac", subset="last_layer")
    else:
        raise Exception("Not implemented")

    #####################################
    #####################################
    #####################################
    
    compressed_X_train = model.compress(X_train).detach()

    model_class_proxy = get_model(str(args.model) + "_proxy")

    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    torch.manual_seed(5)

    
    model_proxy = model_class_proxy(n_classes)
    backup_model_proxy = deepcopy(model_proxy)
    model_proxy = model_proxy.to(DEVICE)
    optimizer_proxy = torch.optim.SGD(
        model_proxy.parameters(),
        lr=args.lr2,
        weight_decay=wd, momentum=mom, nesterov=True)
    for i in tqdm(range(epochs)):
        optimizer_proxy.zero_grad()
        output = model_proxy(compressed_X_train)
        # pred = torch.argmax(output, dim=1)
        # accuracy = torch.sum(pred == y_train)/y_train.shape[0]
        loss = criterion(output, y_train)
        loss.backward()
        optimizer_proxy.step()

    if args.laptype == "full":
        mean1_proxy, prec1_proxy = get_mean_and_prec(compressed_X_train.cpu(), y_train.cpu(), model_proxy, mode="full")
    elif args.laptype =="diag":
        mean1_proxy, prec1_proxy = get_mean_and_prec(compressed_X_train.cpu(), y_train.cpu(), model_proxy, mode="diag")
    elif args.laptype =="kfac":
        mean1_proxy, prec1_proxy = get_mean_and_prec(compressed_X_train.cpu(), y_train.cpu(), model_proxy, mode="kfac")
    else:
        raise Exception("Not implemented")




    kl_seed = []
    kl_seed_ll = []
    idx_seed = []
    pred_seed = []
    kl_seed_proxy = []

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


        compressed_X_train_rem = torch.cat([compressed_X_train[0:i], compressed_X_train[i+1:]])
        compressed_X_train_rem = compressed_X_train_rem.to(DEVICE)

        model_proxy = deepcopy(backup_model_proxy)
        model_proxy = model_proxy.to(DEVICE)
        optimizer_proxy = torch.optim.SGD(
        model_proxy.parameters(),
        lr=args.lr2,
        weight_decay=wd, momentum=mom, nesterov=True)


        for j in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            # pred = torch.argmax(output, dim=1)
            # accuracy = torch.sum(pred == y_train)/y_train.shape[0]
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        for j in range(epochs):
            optimizer_proxy.zero_grad()
            output = model_proxy(compressed_X_train_rem)
            loss = criterion(output, y_train_rem)
            loss.backward()
            optimizer_proxy.step()

        
        if args.laptype == "full":
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="full")
            kl1, square_diff1  = _computeKL_from_full(mean1, mean2, prec1, prec2)
        elif args.laptype =="diag":
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="diag")
            kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
        elif args.laptype =="kfac":
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="kfac")
            kl1, square_diff1 = _computeKL_from_kron(mean1, mean2, prec1, prec2)
        else:
            raise Exception("Not implemented")
        

        if args.laptype == "full":
            mean2_ll, prec2_ll = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="full", subset="last_layer")
            kl1_ll, square_diff1  = _computeKL_from_full(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
        elif args.laptype =="diag":
            mean2_ll, prec2_ll = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="diag", subset="last_layer")
            kl1_ll, square_diff1 = _computeKL(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
        elif args.laptype =="kfac":
            mean2_ll, prec2_ll = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, mode="kfac", subset="last_layer")
            kl1_ll, square_diff1 = _computeKL_from_kron(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
        else:
            raise Exception("Not implemented")
        

        if args.laptype == "full":
            mean2_proxy, prec2_proxy = get_mean_and_prec(compressed_X_train_rem.cpu(), y_train_rem.cpu(), model_proxy, mode="full")
            kl1_proxy, square_diff1  = _computeKL_from_full(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
        elif args.laptype =="diag":
            mean2_proxy, prec2_proxy = get_mean_and_prec(compressed_X_train_rem.cpu(), y_train_rem.cpu(), model_proxy, mode="diag")
            kl1_proxy, square_diff1 = _computeKL(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
        elif args.laptype =="kfac":
            mean2_proxy, prec2_proxy = get_mean_and_prec(compressed_X_train_rem.cpu(), y_train_rem.cpu(), model_proxy, mode="kfac")
            kl1_proxy, square_diff1 = _computeKL_from_kron(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
        else:
            raise Exception("Not implemented")
    

        print(f"KL divergence {kl1}")
        print(f"KL divergence_ll {kl1_ll}")
        print(f"KL divergence_proxy {kl1_proxy}")
        kl_seed.append(kl1)
        kl_seed_ll.append(kl1_ll)
        square_diff.append(square_diff1)
        idx_seed.append(i)
        kl_seed_proxy.append(kl1_proxy)
    kl_proxy.append(kl_seed_proxy)
    kl.append(kl_seed)
    kl_ll.append(kl_seed_ll)
    idx.append(idx_seed)

    result = {"idx": idx,
              "kl": kl,
              "kl_ll": kl_ll,
              "kl_proxy": kl_proxy,
              "pred":pred, 
              "square_diff":square_diff}
    

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')