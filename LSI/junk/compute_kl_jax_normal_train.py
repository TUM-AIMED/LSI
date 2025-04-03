import jax
import jax.numpy as jnp
from Datasets.dataset_helper import get_dataset
import time
import numpy as np
from laplace import Laplace
from LSI.experiments.utils_kl import _computeKL
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


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=3, help="Value for lerining_rate (optional)")
    args = parser.parse_args()


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_normal_train_upd"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_" + str(args.name_ext)
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


    epochs = args.epochs
    alpha = args.lr
    nesterov_momentum = 0.99
    i = 0
    kl = []
    idx = []

    # selu_train_model = jax.jit(train_model)
    start_time = time.time()
    key = jax.random.PRNGKey(0)
    weights = 0.00001 * jax.random.normal(key, shape=(512,10))
    biases = jnp.zeros([10])
    model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum)
    weights_full, bias_full, acc_tr, acc_tes, predictions = model.train_model(epochs, X_train, y_train, alpha)
    print(f"{acc_tr} and {acc_tes}")

      
    result = {"idx": list(range(50000)),
              "predictions": jnp.array(predictions), 
              "label": y_train}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')
    