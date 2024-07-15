import jax
import jax.numpy as jnp
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
from laplace import Laplace
from utils.kl_div import _computeKL, _computeblockKL, _computeKL_from_full
import torch
from torch.utils.data import TensorDataset
from functools import partial
import os
import pickle
import argparse
from models.jax_model import MultinomialLogisticRegressor


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(512, 10)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x

def get_mean_and_prec(data, labels, weights, bias):
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels).to(torch.long)
    data = np.asarray(data)
    data = torch.from_numpy(data).to(torch.float32)
    bias = np.asarray(bias)
    bias = torch.from_numpy(bias)
    weights = np.asarray(weights)
    weights = torch.from_numpy(weights).T.contiguous()
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    tinymodel = TinyModel()

    with torch.no_grad():
        tinymodel.linear1.weight = torch.nn.Parameter(weights)
        tinymodel.linear1.bias = torch.nn.Parameter(bias)

    print(DEVICE)
    la = Laplace(tinymodel.features.to(DEVICE), 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)

    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec

def change_labels(labels, percentage=0.1):
    num_labels = labels.shape[0]
    num_to_change = int(num_labels * percentage)

    # Generate random indices to change
    indices_to_change = jax.random.choice(jax.random.PRNGKey(0), num_labels, shape=(num_to_change,), replace=False)

    # Generate random labels
    random_labels = jax.random.randint(jax.random.PRNGKey(1), minval=0, maxval=10, shape=(num_to_change,))
    for i in range(num_to_change):
        counter = 0
        while random_labels[i] == labels[indices_to_change[i]]:
            counter +=1
            rand_label = jax.random.randint(jax.random.PRNGKey(counter), minval=0, maxval=10, shape=())
            random_labels = random_labels.at[i].set(rand_label)

    # Update the original labels with random labels at selected indices
    return_labels = labels.at[indices_to_change].set(random_labels)

    return return_labels, indices_to_change


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
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="Primacompressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default= 0.01, help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=float, default=0.0)
    args = parser.parse_args()
    print("0")
    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Primacompressed":
        n_classes = 3
        args.subset = 4646
        args.n_rem = 4646

    print("-1")
    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    print("-2")
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)

    X_train = jax.device_put(data_set.data.numpy())
    y_train = jax.device_put(data_set.labels.numpy())
    X_test = jax.device_put(data_set_test.data.numpy())
    y_test = jax.device_put(data_set_test.labels.numpy())
    
    corrupted = []
    y_train, corrupted = change_labels(y_train, percentage=args.corrupt)

    if args.dataset =="cifar10":
        X_train = jnp.ravel(X_train, order="F")


    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    alpha = args.lr
    # nesterov_momentum = 0.99
    # delta = 0.08
    nesterov_momentum = 0.99
    delta = 0.0
    i = 0
    kl = []
    idx = []
    pred = []
    square_diff = []
    # selu_train_model = jax.jit(train_model)
    print("1")
    for seed in range(N_SEEDS):
        start_time = time.time()
        key = jax.random.PRNGKey(seed)
        weights = 0.000001 * jax.random.uniform(key, shape=(512,n_classes))
        biases =  0.000001 * jax.random.uniform(key, shape=([n_classes]))
        model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum)
        print("2")
        weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, X_test, y_test, alpha, delta=delta, batched=args.batch_num)
        # print(f"{acc_tr} and {acc_tes}")
        print("3")
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), weights_full, bias_full)
        # print(f"{time.time() - start_time}")
        kl_seed = []
        idx_seed = []
        pred_seed = []
        for i in range(N_REMOVE):
            model.reset()
            X_train_rm = jnp.delete(X_train, i, axis=0)
            y_train_rm = jnp.delete(y_train, i, axis=0)
            # X_train_rm = X_train
            # y_train_rm = y_train
            start_time2 = time.time()
            weights_rm, bias_rm, acc_tr, acc_tes = model.train_model(epochs, X_train_rm, y_train_rm, X_test, y_test,  alpha, delta=delta, batched=args.batch_num, remove=i)
            print(f"Train took {time.time() - start_time2}")
            start_time2 = time.time()
            mean2, prec2 = get_mean_and_prec(X_train, y_train, weights_rm, bias_rm)
            print(f"KL computation took {time.time() - start_time2}")
            kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
            # kl1, square_diff1 = _computeKL_from_full(mean1, mean2, prec1, prec2)
            print(f"KL divergence {kl1}")
            prediction = model.predict(X_train)
            pred_class = jnp.argmax(prediction, axis=1)
            correct = pred_class == y_train
            kl_seed.append(kl1)
            square_diff.append(square_diff1)
            idx_seed.append(i)
            pred_seed.append(correct)
        kl.append(kl_seed)
        idx.append(idx_seed)
        pred.append(correct)
        # print(f"loop took {time.time() - start_time}")


    result = {"idx": idx,
              "kl": kl,
              "corrupt":corrupted,
              "pred":pred, 
              "square_diff":square_diff}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')