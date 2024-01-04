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

class MultinomialLogisticRegressor():
    def __init__(self, w, b, momentum=0.9):  # Add momentum as an optional parameter
        self.w_init = w
        self.b_init = b
        self.w = w
        self.b = b
        self.momentum = momentum  # Add momentum attribute
        self.w_velocity = jax.tree_map(jnp.zeros_like, w)  # Initialize velocity for weights
        self.b_velocity = jax.tree_map(jnp.zeros_like, b)  # Initialize velocity for biases
        self.grad_fn = jax.grad(self.loss_fn, argnums=(0, 1))

    def reset(self):
        self.w = self.w_init
        self.b = self.b_init
        self.w_velocity = jax.tree_map(jnp.zeros_like, self.w_init)  # Reset velocity for weights
        self.b_velocity = jax.tree_map(jnp.zeros_like, self.b_init)  # Reset velocity for biases

    def predict(self, x):
        return jax.nn.softmax(jax.lax.batch_matmul(x, self.w) + self.b)

    def _predict(self, weights, biases, x):
        return jax.nn.softmax(jax.lax.batch_matmul(x, weights) + biases)

    def cross_entropy(self, logprobs, targets):
        nll = -jnp.take_along_axis(logprobs, targets[:, None], axis=1)
        ce = jnp.mean(nll)
        return ce
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, weights, biases, xs, ys):
        return self.cross_entropy(self._predict(weights, biases, xs), ys) + 0.08 * (
                jnp.mean(weights ** 2) + jnp.mean(biases ** 2))

    def prepare_optim(self, xs, ys, a):
        self.w = jax.device_put(self.w)
        self.b = jax.device_put(self.b)
        self.w_velocity = jax.device_put(self.w_velocity)
        self.b_velocity = jax.device_put(self.b_velocity)
        self.momentum = jax.device_put(self.momentum)
        self.xs = jax.device_put(xs)
        self.ys = jax.device_put(ys)
        self.a = jax.device_put(a)

    def step(self):
        # Compute gradients
        grads = self.grad_fn(self.w - self.momentum, self.b - self.momentum, self.xs, self.ys)
        self.w_velocity = self.momentum * self.w_velocity + self.a * grads[0]
        self.b_velocity = self.momentum * self.b_velocity + self.a * grads[1]
        self.w = jax.device_put(self.w - self.w_velocity)
        self.b = jax.device_put(self.b - self.b_velocity)

    def train_model(self, epochs, xs, ys, alpha):
        self.prepare_optim(X_train, y_train, alpha)
        epochs_arr = jnp.arange(0, epochs, 1)
        predictions = []
        for i in epochs_arr:
            self.step()
            predictions.append(self.predict(xs))
            # if i%200 == 0:
        prediction = self.predict(xs)
        pred_class = jnp.argmax(prediction, axis=1)
        correct1 = jnp.sum(pred_class == ys)

        prediction = self.predict(X_test)
        pred_class = jnp.argmax(prediction, axis=1)
        correct2 = jnp.sum(pred_class == y_test)
        return self.w, self.b, correct1/50000, correct2/10000, predictions

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


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_normal_train"
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
    