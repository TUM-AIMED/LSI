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
    parser.add_argument("--batch_num", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="torch")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--range1", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--range2", type=int, default=100, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default= 0.1, help="Value for lerining_rate (optional)")
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


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2"
    if args.name == None and args.range1 == 0 and args.range2 == 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt) + "_" + str(args.name_ext)
    if args.name == None and args.range1 != 0 or args.range2 != 0:
        file_name = "kl_jax_torch_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_range_" + str(args.range1) + "_" + str(args.range2) + "_corrupt_" + str(args.corrupt) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    X_train = data_set.data.to(DEVICE)
    y_train = data_set.labels.to(DEVICE)
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
            optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.004,
            weight_decay=0.01, momentum=0.9, nesterov=True)
            for i in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_rem)
                # pred = torch.argmax(output, dim=1)
                # accuracy = torch.sum(pred == y_train_rem)/y_train_rem.shape[0]
                # print(accuracy)
                loss = criterion(output, y_train_rem)
                # print(loss.item())
                loss.backward()
                optimizer.step()
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model)
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