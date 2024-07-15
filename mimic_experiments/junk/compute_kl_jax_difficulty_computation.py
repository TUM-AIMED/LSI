import jax
import jax.numpy as jnp
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
from laplace import Laplace
from utils.kl_div import _computeKL
import torch
from torch.utils.data import TensorDataset
from functools import partial
import os
import pickle
import argparse
from copy import deepcopy
from collections import defaultdict
from models.jax_model import MultinomialLogisticRegressor
import random
from tqdm import tqdm
from itertools import chain



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



def get_ordering(path, kind, ytr=None):
    if kind == "random":
        return_list = random.sample(range(50000), 50000)
        return return_list
    if kind == "largest_onion" or kind == "smallest_onion":
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
        idx = final_dict["remove_idx"]
        idx_new = []
        for tup in idx:
            idx_new += list(tup)
        return idx_new
    with open(path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])
    kl_data = np.mean(kl_data, axis=0)
    idx = final_dict["idx"][0]
    sorted_data = sorted(zip(kl_data, idx), key=lambda x: x[0])
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
        for i in np.unique(ytr):
            class_i_idx = [index for index, val in enumerate(ytr) if val == i]
            class_i_order = [index for index in idx if index in class_i_idx]
            classes_ordered.append(class_i_order)
        idx_new = list(chain(*zip(*classes_ordered)))
        return idx_new
    else:
        raise Exception("Not implemented")
    return

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


    # TODO
    smallest_path_onion = ""
    largest_path_onion = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_diff_upd2/kl_jax_epochs_200_n_divs_10_dataset_cifar10compressed_subset_50000_smallest.pkl"
    smallest_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"
    largest_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_difficulty_computation"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_portions_" + str(args.portions) + "_lr_" + str(args.lr) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)

    X_train_init = data_set.data.numpy()
    y_train_init = data_set.labels.numpy()

    # X_train = jax.device_put(data_set.data.numpy())
    # y_train = jax.device_put(data_set.labels.numpy())
    X_test = jax.device_put(data_set_test.data.numpy())
    y_test = jax.device_put(data_set_test.labels.numpy())

    # ordering_smallst_onion = get_ordering(smallest_path_onion, "smallest_onion")
    ordering_largest_onion = get_ordering(largest_path_onion, "largest_onion")
    # ordering_smallst = get_ordering(smallest_path, "smallest")
    ordering_largest = get_ordering(largest_path, "largest")
    ordering_largest_balanced = get_ordering(largest_path, "largest_balanced", ytr=y_train_init)
    random_first_ordering = get_ordering(None, "random")
    orderings = {
        "ordering_largest": ordering_largest,
        "ordering_largest_balanced": ordering_largest_balanced,
        "ordering_largest_onion": ordering_largest_onion,
        "random_first_ordering": random_first_ordering
    }

    epochs = args.epochs
    alpha = args.lr
    nesterov_momentum = 0.99
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

    for ordering_name, order in orderings.items():
        for i in tqdm(range(0, args.portions)):
            portion = 50000 * 1/(args.portions)
            subset_idx = order[int(i*portion):int((i+1)*portion)]

            X_train = jax.device_put(X_train_init[subset_idx])
            y_train = jax.device_put(y_train_init[subset_idx])
            start_time = time.time()
            key = jax.random.PRNGKey(1)
            weights = 0.00001 * jax.random.normal(key, shape=(512,10))
            biases = jnp.zeros([10])
            model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum)

            weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, X_test, y_test, alpha, return_acc_ac_train=True, delta=0)
            # print(f"{time.time() - start_time}")


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