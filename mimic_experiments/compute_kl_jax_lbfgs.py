import jax
from jax.scipy.optimize import minimize
import jax.numpy as jnp
from Datasets.dataset_helper import get_dataset
from copy import deepcopy
from utils.kl_div import _computeKL_from_full
from laplace import Laplace
import torch
from torch.utils.data import TensorDataset
import numpy as np
from utils.kl_div import _computeKL



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
    data = data[:, :-1]
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

    # for _, (data, target, idx, _) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     output = tinymodel(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    la = Laplace(tinymodel.features, 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec

class MultinomialLogisticRegressor_LBFGS():
    def __init__(self, seed):
        self.seed = seed

    def predict(self, x):
        return jax.nn.softmax(jax.lax.batch_matmul(x, self.w))
            
    def _predict(self, theta):
        return jax.nn.softmax(jax.lax.batch_matmul(self.trainx, theta))
    
    def cross_entropy(self, logprobs, targets):
        nll = -jnp.take_along_axis(logprobs, targets[:, None], axis=1)
        ce = jnp.mean(nll)
        return ce
    
    def loss_fn(self, theta):
        theta = theta.reshape([513, -1])
        return self.cross_entropy(self._predict(theta), self.trainy) + 0.8 * jnp.mean(theta**2)
        
    def optimize(self, trainx, trainy):
        self.trainx = trainx
        self.trainy = trainy
        self.w = jnp.zeros((512 + 1,10))
        self.w = jax.random.normal(jax.random.PRNGKey(1), (512 + 1,10))

        theta = self.w.reshape(-1)
        res = minimize(self.loss_fn, theta, method="BFGS", tol=0.00001, options={"maxiter":10000})
        self.w = res.x.reshape(513,10)
        return res.x, res.hess_inv


def append_bias(original_array):
    new_array = jnp.hstack((original_array, jnp.ones((original_array.shape[0], 1))))
    return new_array

data_set_class, data_path = get_dataset("cifar10compressed")

data_set = data_set_class(data_path, train=True)
data_set_test = data_set_class(data_path, train=False) 
X_train = append_bias(jax.device_put(data_set.data.numpy()))
y_train = jax.device_put(data_set.labels.numpy())
X_test = append_bias(jax.device_put(data_set_test.data.numpy()))
y_test = jax.device_put(data_set_test.labels.numpy())

key = jax.random.PRNGKey(123)

model = MultinomialLogisticRegressor_LBFGS(key)

weights_u_bias, _ = model.optimize(X_train, y_train)
weights_u_bias = weights_u_bias.reshape(513,10)
weights = weights_u_bias[:-1, :]
bias = weights_u_bias[-1, :]
mean1, prec1 = get_mean_and_prec(X_train, y_train, weights, bias)
prediction = model.predict(X_train)
pred_class = jnp.argmax(prediction, axis=1)
correct = jnp.sum(pred_class == y_train)
print(f"Accuracy {correct/50000}")

prediction = model.predict(X_test)
pred_class = jnp.argmax(prediction, axis=1)
correct = jnp.sum(pred_class == y_test)
print(f"Accuracy Test {correct/10000}")


i = 2
X_train_rm = jnp.delete(X_train, i, axis=0)
y_train_rm = jnp.delete(y_train, i, axis=0)
weights_u_bias, _ = model.optimize(X_train_rm, y_train_rm)
weights_u_bias = weights_u_bias.reshape(513,10)
weights = weights_u_bias[:-1, :]
bias = weights_u_bias[-1, :]
mean2, prec2 = get_mean_and_prec(X_train, y_train, weights, bias)

kl1 = _computeKL(mean1, mean2, prec1, prec2)
print("")