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
from tqdm import tqdm
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
        batch_size=2048,
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

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_divs", type=int, default=5, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=2, help="Value for lerining_rate (optional)")
    args = parser.parse_args()


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_diff_upd2"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_n_divs_" + str(args.n_divs) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)

    X_train = jax.device_put(data_set.data.numpy())
    y_train = jax.device_put(data_set.labels.numpy())
    X_test = jax.device_put(data_set_test.data.numpy())
    y_test = jax.device_put(data_set_test.labels.numpy())


    N_REMOVE = args.subset
    N_DIVS = args.n_divs
    epochs = args.epochs
    alpha = args.lr
    nesterov_momentum = 0.99
    i = 0
    kl = []
    idx = []
    remove_idx = []
    computed_idx = []
    # selu_train_model = jax.jit(train_model)
    key = jax.random.PRNGKey(42)
    weights = 0.00001 * jax.random.normal(key, shape=(512,10))
    biases = jnp.zeros([10])
    model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum)
    
    keep_indices = [*range(args.subset)]

    active_indices = [*range(args.subset)]

    for iteration_step in tqdm(range(N_DIVS)):
        model.reset()
        X_train = X_train[jnp.array(keep_indices)]
        y_train = y_train[jnp.array(keep_indices)]
        start_time = time.time()
        weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, X_test, y_test, alpha)
        print(f"{acc_tr} and {acc_tes}")
        mean1, prec1 = get_mean_and_prec(X_train, y_train, weights_full, bias_full)
        print(f"{time.time() - start_time}")
        kl_step = []
        idx_step = []
        for i in tqdm(range(y_train.shape[0])):
            model.reset()
            X_train_rm = jnp.delete(X_train, i, axis=0)
            y_train_rm = jnp.delete(y_train, i, axis=0)
            start_time2 = time.time()
            weights_rm, bias_rm, acc_tr, acc_tes = model.train_model(epochs, X_train_rm, y_train_rm, X_test, y_test, alpha)
            print(f"Train took {time.time() - start_time2}")
            start_time2 = time.time()
            mean2, prec2 = get_mean_and_prec(X_train_rm, y_train_rm, weights_rm, bias_rm)
            print(f"KL computation took {time.time() - start_time2}")
            kl1 = _computeKL(mean1, mean2, prec1, prec2)
            print(kl1)
            kl_step.append(kl1)
            idx_step.append(i)
        sorted_lists = sorted(zip(kl_step, idx_step, active_indices), key=lambda x: x[0])
        kl_step, idx_step, active_indices = zip(*sorted_lists)
        kl.append(kl_step)
        idx.append(idx_step)    

        # remove easiest
        if args.name_ext == "smallest":
            keep_indices = idx_step[int(args.subset/N_DIVS):]
            remove_idx.append(active_indices[0:int(args.subset/N_DIVS)])
            computed_idx.append(active_indices)
            active_indices = active_indices[int(args.subset/N_DIVS):]
        elif args.name_ext == "largest":
            # remove the most difficult
            keep_indices = idx_step[0:-int(args.subset/N_DIVS)]
            remove_idx.append(active_indices[-int(args.subset/N_DIVS):])
            computed_idx.append(active_indices)
            active_indices = active_indices[0:-int(args.subset/N_DIVS)]
        else:    
            raise Exception("provide name ext")

        print(f"loop took {time.time() - start_time}")
    result = {"idx": idx,
              "kl": kl,
              "remove_idx": remove_idx,
              "computed_idx": computed_idx}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')