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
from jax import random
from jax import vmap
import itertools


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
    def __init__(self, w, b, momentum=0.9, noise=0, clip=1000):  # Add momentum as an optional parameter
        self.w_init = w
        self.b_init = b
        self.w = w
        self.b = b
        self.momentum = momentum  # Add momentum attribute
        self.w_velocity = jax.tree_map(jnp.zeros_like, w)  # Initialize velocity for weights
        self.b_velocity = jax.tree_map(jnp.zeros_like, b)  # Initialize velocity for biases
        self.grad_fn = jax.jit(jax.grad(self.loss_fn, argnums=(0, 1)))

        self.clip = clip
        self.noise = noise

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

    @partial(jax.jit, static_argnums=(0,))
    def clipped_grad(self, params, l2_norm_clip, single_example_batch_x, single_example_batch_y):
        """Evaluate gradient for a single-example batch and clip its grad norm."""
        single_example_batch_x = jnp.expand_dims(single_example_batch_x, axis=0)
        single_example_batch_y = jnp.expand_dims(single_example_batch_y, axis=0)
        grads = jax.grad(self.loss_fn, argnums=(0, 1))(params[0], params[1], single_example_batch_x, single_example_batch_y)
        nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads[0])
        nonempty_grads_b, tree_def_b = jax.tree_util.tree_flatten(grads[1])
        total_grad_norm = jnp.linalg.norm(
            jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]))
        total_grad_norm_b = jnp.linalg.norm(
            jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads_b]))
        total_total_grad_norm = jnp.linalg.norm(jnp.array([total_grad_norm, total_grad_norm_b]))
        divisor = jnp.maximum(total_total_grad_norm / l2_norm_clip, 1.)
        normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
        normalized_nonempty_grads_b = [g / divisor for g in nonempty_grads_b]
        
        return jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads), jax.tree_util.tree_unflatten(tree_def_b, normalized_nonempty_grads_b)

    @partial(jax.jit, static_argnums=(0,))
    def private_grad(self, params, batch, rng, l2_norm_clip, noise_multiplier,
                    batch_size):
        """Return differentially private gradients for params, evaluated on batch."""
        clipped_grads = vmap(self.clipped_grad, (None, None, 0, 0))(params, l2_norm_clip, batch[0], batch[1])
        clipped_grads_flat, grads_treedef = jax.tree_util.tree_flatten(clipped_grads[0])
        aggregated_clipped_grads = [g.mean(0) for g in clipped_grads_flat]
        noise_stdv = l2_norm_clip * noise_multiplier * 1/batch_size
        # jax.debug.print("{y}", y=random.normal(rng, aggregated_clipped_grads[0].shape))
        normalized_noised_aggregated_clipped_grads = [gx + noise_stdv * random.normal(rng, gx.shape) for gx in aggregated_clipped_grads]
        
        clipped_grads_flat_b, grads_treedef_b = jax.tree_util.tree_flatten(clipped_grads[1])
        aggregated_clipped_grads_b = [g.mean(0) for g in clipped_grads_flat_b]
        normalized_noised_aggregated_clipped_grads_b = [gx + noise_stdv * random.normal(rng, gx.shape) for gx in aggregated_clipped_grads_b]


        return jax.tree_util.tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), jax.tree_util.tree_unflatten(grads_treedef_b, normalized_noised_aggregated_clipped_grads_b)

    def step(self, seed, iteration):
        # Compute gradients
        # grads = self.grad_fn(self.w - self.momentum, self.b - self.momentum, self.xs, self.ys)
        batch_size = len(self.xs)
        grads = self.private_grad((self.w - self.momentum * self.w_velocity, self.b - self.momentum  * self.b_velocity), (self.xs, self.ys), jax.random.fold_in(jax.random.PRNGKey(seed), iteration), self.clip, self.noise, batch_size)
        self.w_velocity = self.momentum * self.w_velocity + self.a * grads[0]
        self.b_velocity = self.momentum * self.b_velocity + self.a * grads[1]
        self.w = jax.device_put(self.w - self.w_velocity)
        self.b = jax.device_put(self.b - self.b_velocity)

    def train_model(self, epochs, xs, ys, xt, yt, alpha, seed):
        self.prepare_optim(xs, ys, alpha)
        epochs_arr = jnp.arange(0, epochs, 1)
        for i in epochs_arr:
            self.step(seed, i)
            if i%200 == 0:
                prediction = self.predict(xs)
                pred_class = jnp.argmax(prediction, axis=1)
                correct1 = jnp.sum(pred_class == ys)
                print(correct1/50000, flush=True)

        prediction = self.predict(xt)
        pred_class = jnp.argmax(prediction, axis=1)
        correct2 = jnp.sum(pred_class == yt)
        return self.w, self.b, correct1/50000, correct2/10000

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
    parser.add_argument("--n_rem", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=700, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar100compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=0.1, help="Value for lerining_rate (optional)")
    parser.add_argument("--clip", type=float, default=0.1, help="Value for lerining_rate (optional)")
    parser.add_argument("--noise", type=float, default=0, help="Value for lerining_rate (optional)")
    args = parser.parse_args()


    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_dp_upd2_fixed_noise_schedule"
    if args.name == None:
        file_name = "kl_jax_epochs_" + str(args.epochs) + "_lr_" + str(args.lr) + "_remove_" + str(args.n_rem) + "_seeds_" + str(args.n_seeds) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_noise_" + str(args.noise) + "_clip_" + str(args.clip) + "_" + str(args.name_ext)
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
    nesterov_momentum = 0.99
    i = 0
    kl = []
    idx = []

    # selu_train_model = jax.jit(train_model)
    for seed in range(N_SEEDS):
        start_time = time.time()
        key = jax.random.PRNGKey(seed)
        weights = 0.00001 * jax.random.uniform(key, shape=(512,100))
        biases = 0.00001 * jax.random.uniform(key, shape=([100]))
        model = MultinomialLogisticRegressor(weights, biases, momentum=nesterov_momentum, noise=args.noise, clip=args.clip)
        weights_full, bias_full, acc_tr, acc_tes = model.train_model(epochs, X_train, y_train, X_test, y_test, alpha, seed)
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
            weights_rm, bias_rm, acc_tr, acc_tes = model.train_model(epochs, X_train_rm, y_train_rm, X_test, y_test, alpha, seed)
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

    # Compose and save results

    result = {"idx": idx,
              "kl": kl,
              "noise": args.noise,
              "clip": args.clip,
              "epochs": args.epochs}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')