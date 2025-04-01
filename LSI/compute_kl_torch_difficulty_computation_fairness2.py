
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
import torch
import os
import pickle
import argparse
import random
from tqdm import tqdm
from copy import deepcopy


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

def get_ordering(paths, kind, len_dataset):
    if kind == "random":
        return_list = random.sample(range(len_dataset), len_dataset)
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
    else:
        raise Exception("Not implemented")
    return


def reduce_classes(combination, target, ordering_smallst, ordering_largest, random_first_ordering, portion):
    # random
    indices_r = []
    indices_s = []
    indices_l = []

    if combination[0] != None:
        class_mask1 = target[random_first_ordering] == combination[0]
        indices_r = np.array(random_first_ordering)[class_mask1]
        indices_r = indices_r[0:int(portion * len(indices_r))]

    # first
    if combination[1] != None:
        class_mask2 = target[ordering_smallst] == combination[1]
        indices_s = np.array(ordering_smallst)[class_mask2]
        indices_s = indices_s[0:int(portion * len(indices_s))]

    # last
    if combination[2] != None:
        class_mask3 = target[ordering_largest] == combination[2]
        indices_l = np.array(ordering_largest)[class_mask3]
        indices_l = indices_l[0:int(portion * len(indices_l))]

    removal_list = np.concatenate((indices_r, indices_s, indices_l))

    index_list = np.array(list(range(len(target))))

    mask = np.isin(index_list, removal_list, invert=True)
    index_list = index_list[mask]
    index_list = list(index_list)

    return index_list

def remove_tuples_with_repeating_values(lst):
    def has_repeating_values(t):
        return len(set(t)) < len(t)
    return [t for t in lst if not has_repeating_values(t)]

def compute_cwa(predictions, labels):
    class_pred = torch.argmax(predictions, axis=1)
    class_accuracy = {}
    for unique_class in torch.unique(labels):
        class_mask = labels == unique_class
        class_accuracy[unique_class.cpu().item()] = (torch.sum(class_pred[class_mask] == unique_class)/sum(class_mask)).cpu().item()
    return class_accuracy

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
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="fairness_")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=0.004, help="Value for lerining_rate (optional)")
    parser.add_argument("--portions", type=int, default=50)
    args = parser.parse_args()

    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Primacompressed":
        n_classes = 3
        args.subset = 4646

    # TODO
    # smallest_path_onion = ""
    # largest_path_onion = ""
    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]

    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]

    paths = [
        "kl_jax_torch_1000_remove_4646_dataset_Primacompressed_subset_4646_range_0_4646_corrupt_0.0_torch.pkl"
    ]
    paths = [base_path + path for path in paths]

    # ordering_smallst_onion = get_ordering(smallest_path_onion, "smallest_onion")
    # ordering_largest_onion = get_ordering(largest_path_onion, "largest_onion")
    ordering_smallst = get_ordering(paths, "smallest", args.subset)
    ordering_largest = get_ordering(paths, "largest", args.subset)
    random_first_ordering = get_ordering(None, "random", args.subset)
   

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_c10_2_kl_torch_fairness_computation_95Per"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_" + str(args.name_ext)
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

    combinations = []
    for i in range(len(np.unique(y_test.cpu().numpy()))):
        combinations.append([i, None, None])
        combinations.append([None, i, None])
        combinations.append([None, None, i])


    epochs = args.epochs
    alpha = args.lr
    nesterov_momentum = 0.99
    i = 0
    kl = []
    idx = []


    idx_train = {}
    train_full_ac = {}
    train_subset_acc = {}
    test_full_acc = {}

    torch.manual_seed(5)

    model = TinyModel(n_classes)
    backup_model = deepcopy(model)

    for combination in tqdm(combinations):
        for i in tqdm(range(1, args.portions)):
            model = deepcopy(backup_model).to(DEVICE)
            portion = i/(args.portions)
            subset_idx = reduce_classes(combination, y_train_init.cpu(), ordering_smallst, ordering_largest, random_first_ordering, portion)
            print(len(subset_idx))
            X_train = X_train_init[subset_idx].to(DEVICE)
            y_train = y_train_init[subset_idx].to(DEVICE)

            start_time = time.time()

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.004,
                weight_decay=0.01, momentum=0.9, nesterov=True)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')

            for i in range(epochs):
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
            acc_tr = compute_acc(model, X_train, y_train)
            acc_tes = compute_acc(model, X_test, y_test)
            print(f"{acc_tr} and {acc_tes}")
            print(f"{time.time() - start_time}")
            prediction_train_init = model(X_train_init)
            prediction_train = model(X_train)
            prediction_test = model(X_test)


            idx_train[str(portion)] = subset_idx
            train_full_ac[str(portion)] = compute_cwa(prediction_train_init, y_train_init)
            train_subset_acc[str(portion)] = compute_cwa(prediction_train, y_train)
            test_full_acc[str(portion)] = compute_cwa(prediction_test, y_test)


       
        result = {"train_acc_subset": acc_tr,
                "test_acc": acc_tes,
                "train_full_acc": train_full_ac,
                "train_subset_acc": train_subset_acc,
                "test_full_acc": test_full_acc,
                "idx_train_subset": idx_train,
                "combination": combination
                }
        

        if not os.path.exists(path_name):
            os.makedirs(path_name)
        file_path = os.path.join(path_name, file_name + str(combination[0]) + "_" + str(combination[1]) + "_" + str(combination[2]) + ".pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(result, file)
        print(f'Saving at {file_path}')
