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
    def __init__(self, w1, b1, w2, b2, momentum=0.9):  # Add momentum as an optional parameter
        self.w1_init = w1
        self.b1_init = b1
        self.w1 = w1
        self.b1 = b1
        self.w2_init = w2
        self.b2_init = b2
        self.w2 = w2
        self.b2 = b2
        self.momentum = momentum  # Add momentum attribute
        self.w1_velocity = jax.tree_map(jnp.zeros_like, w1)  # Initialize velocity for weights
        self.b1_velocity = jax.tree_map(jnp.zeros_like, b1)  # Initialize velocity for biases
        self.w2_velocity = jax.tree_map(jnp.zeros_like, w2)  # Initialize velocity for weights
        self.b2_velocity = jax.tree_map(jnp.zeros_like, b2)  # Initialize velocity for biases
        self.grad_fn = jax.grad(self.loss_fn, argnums=(0, 1, 2, 3))

    def reset(self):
        self.w1 = self.w1_init
        self.b1 = self.b1_init
        self.w1_velocity = jax.tree_map(jnp.zeros_like, self.w1_init)  # Reset velocity for weights
        self.b1_velocity = jax.tree_map(jnp.zeros_like, self.b1_init)  # Reset velocity for biases
        self.w2 = self.w2_init
        self.b2 = self.b2_init
        self.w2_velocity = jax.tree_map(jnp.zeros_like, self.w2_init)  # Reset velocity for weights
        self.b2_velocity = jax.tree_map(jnp.zeros_like, self.b2_init)  # Reset velocity for biases

    def predict(self, x):
        x = jax.nn.relu(jax.lax.batch_matmul(x, self.w1) + self.b1)
        return jax.nn.softmax(jax.lax.batch_matmul(x, self.w2) + self.b2)
        

    def _predict(self, w1, b1, w2, b2, x):
        x = jax.nn.relu(jax.lax.batch_matmul(x, w1) + b1)
        return jax.nn.softmax(jax.lax.batch_matmul(x, w2) + b2)

    def cross_entropy(self, logprobs, targets):
        nll = -jnp.take_along_axis(logprobs, targets[:, None], axis=1)
        ce = jnp.mean(nll)
        return ce
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, w1, b1, w2, b2, xs, ys):
        return self.cross_entropy(self._predict(w1, b1, w2, b2, xs), ys) 

    def prepare_optim(self, xs, ys, a):
        self.w1 = jax.device_put(self.w1)
        self.b1 = jax.device_put(self.b1)
        self.w1_velocity = jax.device_put(self.w1_velocity)
        self.b1_velocity = jax.device_put(self.b1_velocity)
        self.w2 = jax.device_put(self.w2)
        self.b2 = jax.device_put(self.b2)
        self.w2_velocity = jax.device_put(self.w2_velocity)
        self.b2_velocity = jax.device_put(self.b2_velocity)
        self.momentum = jax.device_put(self.momentum)
        self.xs = jax.device_put(xs)
        self.ys = jax.device_put(ys)
        self.a = jax.device_put(a)

    def step(self):
        # Compute gradients
        print(self.loss_fn(self.w1 - self.momentum, self.b1 - self.momentum, self.w2 - self.momentum, self.b2 - self.momentum, self.xs, self.ys))
        grads = self.grad_fn(self.w1 - self.momentum, self.b1 - self.momentum, self.w2 - self.momentum, self.b2 - self.momentum, self.xs, self.ys)
        self.w1_velocity = self.momentum * self.w1_velocity + self.a * grads[0]
        self.b1_velocity = self.momentum * self.b1_velocity + self.a * grads[1]
        self.w2_velocity = self.momentum * self.w2_velocity + self.a * grads[2]
        self.b2_velocity = self.momentum * self.b2_velocity + self.a * grads[3]
        self.w1 = jax.device_put(self.w1 - self.w1_velocity)
        self.b1 = jax.device_put(self.b1 - self.b1_velocity)
        self.w2 = jax.device_put(self.w2 - self.w2_velocity)
        self.b2 = jax.device_put(self.b2 - self.b2_velocity)

    def train_model(self, epochs, xs, ys, alpha):
        self.prepare_optim(X_train, y_train, alpha)
        epochs_arr = jnp.arange(0, epochs, 1)
        for i in epochs_arr:
            self.step()
            if i%2 == 0:
                prediction = self.predict(xs)
                pred_class = jnp.argmax(prediction, axis=1)
                correct1 = jnp.sum(pred_class == ys)
                # print(correct1/50000)

        prediction = self.predict(X_test)
        pred_class = jnp.argmax(prediction, axis=1)
        correct2 = jnp.sum(pred_class == y_test)
        return self.w1, self.b1, correct1/50000, correct2/10000

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
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem", type=int, default=2, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=0.1, help="Value for lerining_rate (optional)")
    args = parser.parse_args()


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_" + str(args.name_ext)
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

    if args.dataset =="cifar10":
        X_train = jnp.ravel(X_train, order="F")


    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    alpha = args.lr
    nesterov_momentum = 0.5
    i = 0
    kl = []
    idx = []

    # selu_train_model = jax.jit(train_model)
    for seed in range(N_SEEDS):
        start_time = time.time()
        key = jax.random.PRNGKey(seed)
        weights1 = 0.001 * jax.random.normal(key, shape=(512,100))
        biases1 = jnp.zeros([100])
        weights2 = 0.001 * jax.random.normal(key, shape=(100,10))
        biases2 = jnp.zeros([10])
        model = MultinomialLogisticRegressor(weights1, biases1, weights2, biases2, momentum=nesterov_momentum)
        weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, alpha)
        print(f"{acc_tr} and {acc_tes}")
        mean1, prec1 = get_mean_and_prec(X_train, y_train, weights_full, bias_full)
        print(f"{time.time() - start_time}")
        kl_seed = []
        idx_seed = []
        for i in range(N_REMOVE):
            model.reset()
            X_train_rm = jnp.delete(X_train, i, axis=0)
            y_train_rm = jnp.delete(y_train, i, axis=0)
            start_time2 = time.time()
            weights_rm, bias_rm, acc_tr, acc_tes = model.train_model(epochs, X_train_rm, y_train_rm, alpha)
            print(f"Train took {time.time() - start_time2}")
            start_time2 = time.time()
            mean2, prec2 = get_mean_and_prec(X_train_rm, y_train_rm, weights_rm, bias_rm)
            print(f"KL computation took {time.time() - start_time2}")
            kl1 = _computeKL(mean1, mean2, prec1, prec2)
            kl_seed.append(kl1)
            idx_seed.append(i)
        kl.append(kl_seed)
        idx.append(idx_seed)
        print(f"loop took {time.time() - start_time}")
    result = {"idx": idx,
              "kl": kl}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')