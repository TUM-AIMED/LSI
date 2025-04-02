import os
import time
import pickle
import random
import argparse
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from Datasets.dataset_helper import get_dataset
from LSI.models.models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TinyModel, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(512, n_classes))

    def forward(self, x):
        return self.features(x.to(torch.float32))

def get_ordering(file_paths, ordering_type, labels=None, dataset_length=50000):
    """
    Generate an ordering of dataset indices based on KL divergence or random sampling.

    Args:
        file_paths (list): List of file paths containing KL divergence data.
        ordering_type (str): Type of ordering ('random', 'largest', 'largest_balanced', 'smallest').
        labels (torch.Tensor, optional): Labels of the dataset, required for 'largest_balanced'.
        dataset_length (int): Total number of samples in the dataset.

    Returns:
        list: Ordered list of dataset indices.
    """
    if ordering_type == "random":
        return random.sample(range(dataset_length), dataset_length)

    indices, kl_values = [], []
    for batch_idx, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            kl_data = np.mean(pickle.load(file)["kl"], axis=0)
            indices.extend([i + batch_idx * len(kl_data) for i in range(len(kl_data))])
            kl_values.extend(kl_data)

    # Extend KL values and indices to match the dataset length
    while len(kl_values) < dataset_length:
        kl_values.append(kl_values[-1])
        indices.append(indices[-1] + 1)

    # Sort indices based on KL values
    sorted_data = sorted(zip(kl_values, indices), key=lambda x: x[0])
    sorted_indices = [item[1] for item in sorted_data]

    if ordering_type == "largest":
        return sorted_indices[::-1]
    elif ordering_type == "largest_balanced":
        sorted_indices.reverse()
        class_orderings = [
            [idx for idx in sorted_indices if labels[idx] == class_label]
            for class_label in torch.unique(labels)
        ]
        initial_class_lengths = [len(class_indices) for class_indices in class_orderings]
        balanced_indices = []

        while any(class_orderings):
            class_ratios = [
                len(class_indices) / initial_length if class_indices else 0
                for class_indices, initial_length in zip(class_orderings, initial_class_lengths)
            ]
            largest_class_idx = class_ratios.index(max(class_ratios))
            balanced_indices.append(class_orderings[largest_class_idx].pop(0))

        return balanced_indices
    elif ordering_type == "smallest":
        return sorted_indices
    else:
        raise ValueError(f"Ordering type '{ordering_type}' is not implemented.")

def compute_acc(model, inputs, labels):
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item() / labels.size(0)

def load_dataset(args):
    if args.dataset.endswith("_smaller"):
        dataset_name = args.dataset.split("_")[0]
        data_set_class, data_path = get_dataset(dataset_name)
        train_set = data_set_class(data_path, train=True, ret4=False, smaller=True)
        test_set = data_set_class(data_path, train=False, ret4=False, smaller=True)
    else:
        data_set_class, data_path = get_dataset(args.dataset)
        train_set = data_set_class(data_path, train=True, ret4=False)
        test_set = data_set_class(data_path, train=False, ret4=False)
    return train_set, test_set

def main():
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rem", type=int, default=2)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--name_ext", type=str, default="4orders_lrscheduler")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="Imagenette")
    parser.add_argument("--model", type=str, default="ResNet9")
    parser.add_argument("--subset", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--portions", type=int, default=3)
    args = parser.parse_args()

    dataset_paths = {
        "cifar10": [
            "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
            "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
            "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
            "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
            "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl",
        ],
        "Imagenette": [
            "kl_jax_torch_1000_remove_1000_dataset_Imagenettecompressed_subset_9469_range_0_9469_corrupt_0.1_torch.pkl"
        ],
    }

    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    paths = [os.path.join(base_path, p) for p in dataset_paths.get(args.dataset, [])]

    train_set, test_set = load_dataset(args)
    X_train_init, y_train_init = train_set.data.to(DEVICE), train_set.labels.to(DEVICE)
    X_test, y_test = test_set.data.to(DEVICE), test_set.labels.to(DEVICE)

    ordering = get_ordering(paths, "largest_balanced", ytr=y_train_init, len_dataset=args.subset)
    orderings = {"ordering_largest_balanced": ordering}

    model_class = TinyModel if args.model == "Tinymodel" else get_model(args.model)
    backup_model = model_class(n_classes=len(torch.unique(y_train_init)))

    results_train, results_test = defaultdict(dict), defaultdict(dict)

    for ordering_name, order in orderings.items():
        for i in tqdm(range(args.portions)):
            model = deepcopy(backup_model).to(DEVICE)
            portion = args.subset // args.portions
            subset_idx = order[i * portion : (i + 1) * portion]
            X_train, y_train = X_train_init[subset_idx], y_train_init[subset_idx]

            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=0.01, momentum=0.9, nesterov=True
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 4, gamma=0.6)
            criterion = torch.nn.CrossEntropyLoss()

            acc_train, acc_test = [], []
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 10 == 0:
                    acc_train.append(compute_acc(model, X_train, y_train))
                    acc_test.append(compute_acc(model, X_test, y_test))

            results_train[ordering_name][i] = acc_train
            results_test[ordering_name][i] = acc_test

    result = {
        "train_acc_subset": results_train,
        "test_acc": results_test,
        "train_unique": torch.max(y_train_init.unique(return_counts=True)[1]) / len(y_train_init),
        "test_unique": torch.max(y_test.unique(return_counts=True)[1]) / len(y_test),
    }

    output_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_torch_difficulty_computation_after_workshop"
    os.makedirs(output_dir, exist_ok=True)
    file_name = args.name or f"kl_jax_epochs_{args.epochs}_remove_{args.n_rem}_dataset_{args.dataset}_model_{args.model}_portions_{args.portions}_lr_{args.lr}_{args.name_ext}.pkl"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f"Results saved at {file_path}")

if __name__ == "__main__":
    main()
