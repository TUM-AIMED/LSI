import jax
import jax.numpy as jnp
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
import torch
import os
import pickle
import argparse
from collections import defaultdict
from itertools import product
from models.jax_model import MultinomialLogisticRegressor
import random
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def get_ordering(path, kind):
    if kind == "random":
        return_list = random.sample(range(50000), 50000)
        return return_list
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
    else:
        raise Exception("Not implemented")
    return


def reduce_classes(tp, target, ordering_smallst, ordering_largest, random_first_ordering, portion):
    # random
    indices_r = []

    if tp == "low": 
        ordering = ordering_smallst
    elif tp == "high":
        ordering = ordering_largest
    elif tp == "rand":
        ordering = random_first_ordering
       
    for label in np.unique(target):
        class_mask1 = target[ordering] == label
        indices = np.array(ordering)[class_mask1]
        indices = indices[0:int(portion)]
        indices_r.append(indices)

    removal_list = np.concatenate(indices_r)

    index_list = np.array(list(range(50000)))

    mask = np.isin(index_list, removal_list, invert=True)
    index_list = index_list[mask]
    index_list = list(index_list)

    return index_list

def remove_tuples_with_repeating_values(lst):
    def has_repeating_values(t):
        return len(set(t)) < len(t)
    return [t for t in lst if not has_repeating_values(t)]

def compute_cwa(predictions, labels):
    class_pred = np.argmax(predictions, axis=1)
    class_accuracy = {}
    for unique_class in np.unique(labels):
        class_mask = labels == unique_class
        class_accuracy[unique_class] = np.sum(class_pred[class_mask] == unique_class)/sum(class_mask)
    return class_accuracy


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem", type=int, default=2, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="summarization")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=2, help="Value for lerining_rate (optional)")
    parser.add_argument("--portions", type=int, default=100)
    args = parser.parse_args()


    # TODO
    # smallest_path_onion = ""
    # largest_path_onion = ""
    smallest_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"
    largest_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"

    # ordering_smallst_onion = get_ordering(smallest_path_onion, "smallest_onion")
    # ordering_largest_onion = get_ordering(largest_path_onion, "largest_onion")
    ordering_smallst = get_ordering(smallest_path, "smallest")
    ordering_largest = get_ordering(largest_path, "largest")
    random_first_ordering = get_ordering(None, "random")
   

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_summarization/"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_" + str(args.name_ext)
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

    combinations = []
    for i in range(len(np.unique(y_test))):
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

    key = jax.random.PRNGKey(1)
    weights = 0.00001 * jax.random.normal(key, shape=(512,10))
    biases = jnp.zeros([10])
    model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum)
    result_all = []
    for tp in ["low", "high", "rand"]:
        idx_train = {}
        train_full_ac = {}
        train_subset_acc = {}
        test_full_acc = {}
        for i in tqdm(range(1, args.portions)):
            model.reset()
            portion = 5000 * i/(args.portions)
            subset_idx = reduce_classes(tp, y_train_init, ordering_smallst, ordering_largest, random_first_ordering, portion)
            print(sum(subset_idx))
            X_train = jax.device_put(X_train_init[subset_idx])
            y_train = jax.device_put(y_train_init[subset_idx])
            start_time = time.time()
            weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, X_test, y_test, alpha)
            print(f"{acc_tr} and {acc_tes}")
            print(f"{time.time() - start_time}")
            prediction_train_init = model.predict(X_train_init)
            prediction_train = model.predict(X_train)
            prediction_test = model.predict(X_test)


            idx_train[str(portion)] = subset_idx
            train_full_ac[str(portion)] = compute_cwa(prediction_train_init, y_train_init)
            train_subset_acc[str(portion)] = compute_cwa(prediction_train, y_train)
            test_full_acc[str(portion)] = compute_cwa(prediction_test, y_test)
       
        result = {
                "train_full_acc": train_full_ac,
                "train_subset_acc": train_subset_acc,
                "test_full_acc": test_full_acc,
                "idx_train_subset": idx_train,
                "combination": tp
                }
        result_all.append(result)

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result_all, file)
    print(f'Saving at {file_path}')

        # results_performance_train = defaultdict(dict)
        # results_performance_test = defaultdict(dict)
        # pred_train_init = defaultdict(dict)
        # pred_train = defaultdict(dict)
        # lab_train_init = defaultdict(dict)
        # lab_train = defaultdict(dict)
        # idx_train = defaultdict(dict)
        # pred_test = defaultdict(dict)