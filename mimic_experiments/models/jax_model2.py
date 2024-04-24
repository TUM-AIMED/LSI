
import jax
import jax.numpy as jnp
from functools import partial



class MultinomialLogisticRegressor():
    def __init__(self, w1, b1, w2, b2, momentum=0.9):  # Add momentum as an optional parameter
        self.w_init1 = w1
        self.b_init1 = b1
        self.w_init2 = w2
        self.b_init2 = b2
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.momentum = momentum  # Add momentum attribute
        self.w_velocity1 = jax.tree_map(jnp.zeros_like, w1)  # Initialize velocity for weights
        self.b_velocity1 = jax.tree_map(jnp.zeros_like, b1)  # Initialize velocity for biases
        self.w_velocity2 = jax.tree_map(jnp.zeros_like, w2)  # Initialize velocity for weights
        self.b_velocity2 = jax.tree_map(jnp.zeros_like, b2)  # Initialize velocity for biases
        self.grad_fn = jax.jit(jax.grad(self.loss_fn, argnums=(0, 1, 2, 3)))

    def reset(self):
        self.w1 = self.w_init1
        self.b1 = self.b_init1
        self.w_velocity1 = jax.tree_map(jnp.zeros_like, self.w_init1)  # Reset velocity for weights
        self.b_velocity1 = jax.tree_map(jnp.zeros_like, self.b_init1)  # Reset velocity for biases
        self.w2 = self.w_init2
        self.b2 = self.b_init2
        self.w_velocity2 = jax.tree_map(jnp.zeros_like, self.w_init2)  # Reset velocity for weights
        self.b_velocity2 = jax.tree_map(jnp.zeros_like, self.b_init2)  # Reset velocity for biases

    def predict(self, x):
        x = jax.nn.relu(jax.lax.batch_matmul(x, self.w1) + self.b1)
        return jax.nn.softmax(jax.lax.batch_matmul(x, self.w2) + self.b2)

    def _predict(self, weights1, biases1,  weights2, biases2, x):
        # jax.debug.print("{x}", x=x[0:10, 0:10])
        x = jax.nn.relu(jax.lax.batch_matmul(x, weights1) + biases1)
        # jax.debug.print("{x}", x=x[0:10, 0:10])
        x = jax.nn.softmax(jax.lax.batch_matmul(x, weights2) + biases2)
        # jax.debug.print("{x}", x=x[0:10, 0:10])
        return x

    def cross_entropy(self, logprobs, targets):
        nll = -jnp.take_along_axis(logprobs, targets[:, None], axis=1)
        ce = jnp.mean(nll)
        # jax.debug.print("{x}", x=ce)
        # jax.debug.print("{x}", x=logprobs[0:10, :])
        return ce
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, weights1, biases1, weights2, biases2, xs, ys):
        return self.cross_entropy(self._predict(weights1, biases1, weights2, biases2, xs), ys) # + 0.04 * (
                # jnp.mean(weights1 ** 2) + jnp.mean(biases1 ** 2) + jnp.mean(weights2 ** 2) + jnp.mean(biases2 ** 2))

    def prepare_optim(self, xs, ys, a):
        self.w1 = jax.device_put(self.w1)
        self.b1 = jax.device_put(self.b1)
        self.w_velocity1 = jax.device_put(self.w_velocity1)
        self.b_velocity1 = jax.device_put(self.b_velocity1)
        self.w2 = jax.device_put(self.w2)
        self.b2 = jax.device_put(self.b2)
        self.w_velocity2 = jax.device_put(self.w_velocity2)
        self.b_velocity2 = jax.device_put(self.b_velocity2)
        self.momentum = jax.device_put(self.momentum)
        self.xs = jax.device_put(xs)
        self.ys = jax.device_put(ys)
        self.a = jax.device_put(a)

    def step(self):
        # Compute gradients
        grads = self.grad_fn(self.w1 - self.momentum  * self.w_velocity1, 
                             self.b1 - self.momentum * self.b_velocity1,
                             self.w2 - self.momentum  * self.w_velocity2, 
                             self.b2 - self.momentum * self.b_velocity2,
                             self.xs, self.ys)
        self.w_velocity1 = self.momentum * self.w_velocity1 + self.a * grads[0]
        self.b_velocity1 = self.momentum * self.b_velocity1 + self.a * grads[1]
        self.w_velocity2 = self.momentum * self.w_velocity2 + self.a * grads[2]
        self.b_velocity2 = self.momentum * self.b_velocity2 + self.a * grads[3]
        self.w1 = jax.device_put(self.w1 - self.w_velocity1)
        self.b1 = jax.device_put(self.b1 - self.b_velocity1)
        self.w2 = jax.device_put(self.w2 - self.w_velocity2)
        self.b2 = jax.device_put(self.b2 - self.b_velocity2)

    def train_model(self, epochs, xs, ys, xt, yt, alpha, return_acc_ac_train=False):
        self.prepare_optim(xs, ys, alpha)
        epochs_arr = jnp.arange(0, epochs, 1)
        cor1 = []
        cor2 = []
        for i in epochs_arr:
            self.step()
            if return_acc_ac_train: 
                prediction = self.predict(xs)
                pred_class = jnp.argmax(prediction, axis=1)
                correct1 = jnp.sum(pred_class == ys)/xs.shape[0]

                prediction = self.predict(xt)
                pred_class = jnp.argmax(prediction, axis=1)
                correct2 = jnp.sum(pred_class == yt)/xt.shape[0]
                cor1.append(correct1)
                cor2.append(correct2)
                print(f"train_acc {correct1}, test_acc {correct2}")
        if return_acc_ac_train:
            return self.w1, self.b1, cor1, cor2
        return self.w1, self.b1, correct1, correct2