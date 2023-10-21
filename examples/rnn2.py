"""
relu rnn with a forced diagonal recurrence matrix
uses MNIST columns as data

run with OPT=0
"""

import os
import time
import numpy as np
from extra.datasets import fetch_mnist
from tinygrad.helpers import dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.ops import GlobalCounters
from tinygrad.tensor import Tensor

STEPS = int(os.getenv('GRAPH', '2000'))

class RNN:
  def __init__(self):
    self.A = Tensor.ones(128)
    self.B = Tensor.randn(28, 128) * 0.001
    self.C = Tensor.randn(128, 10) * 0.001

  def __call__(self, us):
    us = us.reshape(-1, 28, 28)
    x = Tensor.zeros(us.shape[0], 128)
    for col in range(28):
      u = us[:,:,col]
      x = x.mul(self.A) + u.linear(self.B)
      x = x.relu()
    return x.linear(self.C)


rnn = RNN()
opt = SGD(get_parameters(rnn), lr=1e-5)

X_train, Y_train, X_test, Y_test = fetch_mnist()

def evaluate(rnn):
  @TinyJit
  def jit(x):
   return rnn(x).log_softmax().realize()

  avg_acc = 0
  for step in range(0, len(X_test), 16):
    samp = np.arange(step, min(step+16, len(X_test)))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass with jit
    logits = jit(batch)

    # calculate accuracy
    pred = logits.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).sum()

  return avg_acc / len(X_test)

with Tensor.train():
  st = time.monotonic()
  for step in range(STEPS):
    GlobalCounters.reset()

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
    loss_cpu = loss.numpy()

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = logits.argmax(axis=-1)
    acc = (pred == labels).mean().numpy()
    cl = time.monotonic()

    if step % 100 == 99:
      ops = GlobalCounters.global_ops
      eval_acc = evaluate(rnn)
      ev = time.monotonic()
      print(f"{step:4d} {(cl-st)*1000.0:7.2f} ms run, {loss_cpu:7.2f} loss, {acc:2.3f} accuracy, {(ev-cl)*1000.0:7.2f} ms eval, {eval_acc:2.3f} eval accuracy, {opt.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {ops*1e-9/(cl-st):9.2f} GFLOPS")

    st = cl


state_dict = get_state_dict(rnn)
safe_save(state_dict, "model.safetensors")
state_dict = safe_load("model.safetensors")
load_state_dict(rnn, state_dict)
print("final evaluation accuracy: ", evaluate(rnn))

# DEBUG=1
# GRAPH=1
# open -a /Application/Safari.app /tmp/net.svg
