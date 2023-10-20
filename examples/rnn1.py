"""
relu rnn with a diagonal recurrence matrix, following https://arxiv.org/abs/1504.00941
uses MNIST columns as data
"""

import numpy as np
from extra.datasets import fetch_mnist
from tinygrad.helpers import Timing
from tinygrad.helpers import dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.tensor import Tensor


class RNN:
  def __init__(self):
    self.A = Tensor.eye(128)
    self.B = Tensor.randn(28, 128) * 0.001
    self.C = Tensor.randn(128, 10) * 0.001

  def __call__(self, us):
    us = us.reshape(-1, 28, 28)
    x = Tensor.zeros(us.shape[0], 128)
    for col in range(28):
      u = us[:,:,col]
      x = x.linear(self.A)
      x = x + u.linear(self.B)
      x = x.relu()
    return x.linear(self.C)


rnn = RNN()
opt = SGD(get_parameters(rnn), lr=1e-5)

X_train, Y_train, X_test, Y_test = fetch_mnist()

with Tensor.train():
  for step in range(2000):
    # random sample a batch
    samp = np.random.randint(0, X_train.shape[0], size=(64))
    batch = Tensor(X_train[samp], requires_grad=False)
    # get the corresponding labels
    labels = Tensor(Y_train[samp])

    # forward pass
    logits = rnn(batch)

    # compute loss
    y_counter = Tensor.arange(10, dtype=dtypes.int32, requires_grad=False)[None, :].expand(labels.numel(), logits.shape[-1])
    y = ((y_counter == labels.flatten().reshape(-1, 1)).where(-1.0, 0)).reshape(*labels.shape, logits.shape[-1])

    loss = logits.log_softmax().mul(y).sum()

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = logits.argmax(axis=-1)
    acc = (pred == labels).mean()

    if step % 100 == 0:
      print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


state_dict = get_state_dict(rnn)
safe_save(state_dict, "model.safetensors")
state_dict = safe_load("model.safetensors")
load_state_dict(rnn, state_dict)

@TinyJit
def jit(x):
  return rnn(x).realize()

with Timing("Time: "):
  avg_acc = 0
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass with jit
    logits = jit(batch)

    # calculate accuracy
    pred = logits.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / 1000}")


# DEBUG=1
# GRAPH=1
# open -a /Application/Safari.app /tmp/net.svg
