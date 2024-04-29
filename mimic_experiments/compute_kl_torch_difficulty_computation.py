
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
import torch
import os
import pickle
import argparse
from copy import deepcopy
from collections import defaultdict
import random
from tqdm import tqdm
from itertools import chain



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


def get_ordering(paths, kind, ytr=None, len_dataset=50000):
    if kind == "random":
        return_list = random.sample(range(50000), 50000)
        return return_list
    idxs = []
    kls = []
    for batch, path in enumerate(paths):
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
            kl_data = np.array(final_dict["kl"])
            kl_data = np.mean(kl_data, axis=0)
            idx = list(range(len(kl_data)))
            idx = [idx_i + 10000 * batch for idx_i in idx]
            kls.extend(kl_data)
            idxs.extend(idx)
    while len(kls) < len_dataset:
        kls.append(kls[-1])
        idxs.append(idxs[-1] + 1)
    sorted_data = sorted(zip(kls, idxs), key=lambda x: x[0])
    kl_data, idx = zip(*sorted_data)
    idx = list(idx)
    if kind == "smallest":
        return idx
    elif kind == "largest":
        idx.reverse()
        return idx
    elif kind== "largest_balanced":
        idx.reverse()
        classes_ordered = []
        for i in torch.unique(ytr):
            class_i_idx = [index for index, val in enumerate(ytr) if val == i]
            class_i_order = [index for index in idx if index in class_i_idx]
            classes_ordered.append(class_i_order)
        idx_new = list(chain(*zip(*classes_ordered)))
        return idx_new
    else:
        raise Exception("Not implemented")
    return

def compute_acc(model, inputs, labels):
    correct = 0
    total = 0
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct/total

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem", type=int, default=2, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="4orders")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=0.02, help="Value for lerining_rate (optional)")
    parser.add_argument("--portions", type=int, default=20)
    args = parser.parse_args()

    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Primacompressed":
        n_classes = 3

    # TODO
    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    paths = [
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    ]

    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]

    paths = [base_path + path for path in paths]

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_torch_difficulty_computation"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_portions_" + str(args.portions) + "_lr_" + str(args.lr) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)

    X_train_init = data_set.data.to(DEVICE)
    y_train_init = data_set.labels.to(DEVICE)

    X_train = data_set.data.to(DEVICE)
    y_train = data_set.labels.to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)

    # ordering_smallst_onion = get_ordering(smallest_path_onion, "smallest_onion")
    # ordering_largest_onion = get_ordering(largest_path_onion, "largest_onion")
    ordering_smallst = get_ordering(paths, "smallest")
    ordering_largest = get_ordering(paths, "largest")
    ordering_largest_balanced = get_ordering(paths, "largest_balanced", ytr=y_train_init)
    random_first_ordering = get_ordering(None, "random")
    orderings = {
        "ordering_largest": ordering_largest,
        "ordering_largest_balanced": ordering_largest_balanced,
        "random_first_ordering": random_first_ordering,
        "ordering_smallest": ordering_smallst
    }

    epochs = args.epochs
    i = 0
    kl = []
    idx = []

    results_performance_train = defaultdict(dict)
    results_performance_test = defaultdict(dict)
    pred_train_init = defaultdict(dict)
    pred_train = defaultdict(dict)
    lab_train_init = defaultdict(dict)
    lab_train = defaultdict(dict)
    idx_train = defaultdict(dict)
    pred_test = defaultdict(dict)

    torch.manual_seed(5)

    model = TinyModel(n_classes)
    backup_model = deepcopy(model)

    for ordering_name, order in orderings.items():
        for i in tqdm(range(0, args.portions)):
            model = deepcopy(backup_model).to(DEVICE)
            portion = 50000 * 1/(args.portions)
            subset_idx = order[int(i*portion):int((i+1)*portion)]

            X_train = X_train_init[subset_idx]
            y_train =y_train_init[subset_idx]

            start_time = time.time()

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.004,
                weight_decay=0.01, momentum=0.9, nesterov=True)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            acc_tr  = []
            acc_tes = []
            for j in range(epochs):
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
                acc_tr.append(compute_acc(model, X_train, y_train))
                acc_tes.append(compute_acc(model, X_test, y_test))


            results_performance_train[ordering_name][i] = acc_tr
            results_performance_test[ordering_name][i] = acc_tes

       
    result = {"train_acc_subset": results_performance_train,
              "test_acc": results_performance_test
              }
    

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')