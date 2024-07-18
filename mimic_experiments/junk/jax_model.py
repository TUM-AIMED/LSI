
import jax
import jax.numpy as jnp
from functools import partial
import math


class MultinomialLogisticRegressor():
    def __init__(self, w, b, momentum=0.9):  # Add momentum as an optional parameter
        self.w_init = w
        self.b_init = b
        self.w = w
        self.b = b
        self.momentum = momentum  # Add momentum attribute
        self.w_velocity = jax.tree_map(jnp.zeros_like, w)  # Initialize velocity for weights
        self.b_velocity = jax.tree_map(jnp.zeros_like, b)  # Initialize velocity for biases
        self.grad_fn = jax.jit(jax.grad(self.loss_fn, argnums=(0, 1)))

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
        return self.cross_entropy(self._predict(weights, biases, xs), ys) + self.delta * (
                jnp.mean(weights ** 2) + jnp.mean(biases ** 2))

    def prepare_optim(self, xs, ys, a, delta):
        self.w = jax.device_put(self.w)
        self.b = jax.device_put(self.b)
        self.w_velocity = jax.device_put(self.w_velocity)
        self.b_velocity = jax.device_put(self.b_velocity)
        self.momentum = jax.device_put(self.momentum)
        self.xs = jax.device_put(xs)
        self.ys = jax.device_put(ys)
        self.a = jax.device_put(a)
        self.delta = jax.device_put(delta)

    def step(self, batch, remove):
        # Compute gradients
        if self.max_batch == 1:
            train_xs = self.xs
            train_ys = self.ys
        else:                
            batchsize = math.ceil(self.xs.shape[0]/self.max_batch)
            if remove != None:
                removedbatch = remove%batchsize
                if removedbatch == batch:
                    train_xs = self.xs[batch * batchsize:(batch+1)*batchsize - 1]
                    train_ys = self.ys[batch * batchsize:(batch+1)*batchsize - 1]
                elif removedbatch > batch:
                    train_xs = self.xs[batch * batchsize:(batch+1)*batchsize]
                    train_ys = self.ys[batch * batchsize:(batch+1)*batchsize]
                elif removedbatch < batch:
                    train_xs = self.xs[batch * batchsize - 1:(batch+1)*batchsize - 1]
                    train_ys = self.ys[batch * batchsize - 1:(batch+1)*batchsize - 1]
            else:
                train_xs = self.xs[batch * batchsize:(batch+1)*batchsize]
                train_ys = self.ys[batch * batchsize:(batch+1)*batchsize]

        grads = self.grad_fn(self.w - self.momentum*self.w_velocity, self.b - self.momentum*self.b_velocity, train_xs, train_ys)
        loss = self.loss_fn(self.w - self.momentum*self.w_velocity, self.b - self.momentum*self.b_velocity, train_xs, train_ys)
        self.w_velocity = self.momentum * self.w_velocity + self.a * grads[0]
        self.b_velocity = self.momentum * self.b_velocity + self.a * grads[1]
        self.w = jax.device_put(self.w - self.w_velocity)
        self.b = jax.device_put(self.b - self.b_velocity)
        return loss

    def train_model(self, epochs, xs, ys, xt, yt, alpha, return_acc_ac_train=False, delta=0.08, batched=1, remove=None):
        self.max_batch = batched
        self.prepare_optim(xs, ys, alpha, delta)
        epochs_arr = jnp.arange(0, epochs, 1)
        cor1 = []
        cor2 = []
        for i in epochs_arr:
            for j in range(batched):
                loss = self.step(j, remove)
            if i%200 == 0:
                prediction = self.predict(xs)
                pred_class = jnp.argmax(prediction, axis=1)
                correct1 = jnp.sum(pred_class == ys)/xs.shape[0]
                print(correct1)
                print(loss)
            if return_acc_ac_train: 
                prediction = self.predict(xs)
                pred_class = jnp.argmax(prediction, axis=1)
                correct1 = jnp.sum(pred_class == ys)/xs.shape[0]

                prediction = self.predict(xt)
                pred_class = jnp.argmax(prediction, axis=1)
                correct2 = jnp.sum(pred_class == yt)/xt.shape[0]
                cor1.append(correct1)
                cor2.append(correct2)

        prediction = self.predict(xs)
        pred_class = jnp.argmax(prediction, axis=1)
        correct1 = jnp.sum(pred_class == ys)/xs.shape[0]

        prediction = self.predict(xt)
        pred_class = jnp.argmax(prediction, axis=1)
        correct2 = jnp.sum(pred_class == yt)/xt.shape[0]
        
        if return_acc_ac_train:
            return self.w, self.b, cor1, cor2
        return self.w, self.b, correct1, correct2