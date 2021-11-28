
---
layout: post
title: Online EWC
description: with no page entry here
redirect: https://unsplash.com
---

Motivation of Online EWC
============

In real applications, offline EWC can be costly in case of a large number of tasks and
will become more and more expensive as the task number grows. This is because offline EWC
tends to store the fisher information matrix of every task trained before and this can be
huge with a large number of tasks. This is where online EWC comes to the rescue.

How Online EWC works
--------------

Online EWC achieves a constant and lower cost by always maintaining one fisher information matrix. 
Each time a new task is trained, the fisher information matrix is updated using a given weight.
Assume the old fisher information matrix is F, the new fisher information matrix is F', and 
the weight of the new task is . Mathematically, the update process can be expressed as

< img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"><img/>
![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)



Implementation of Online EWC
--------------

To implement the online EWC method, we first define the online EWC class using pytorch

~~~python
class OnlineEWC:
  def __init__(self, model: nn.Module, loss=nn.MSELoss()):
    self._model = model
    self._params = {}
    self._fim = {}
    self._loss = loss
    self._optim = None
    self._lambda = 0

  def train(self, inputs, labels, lr, alpha = 0.5, lam=0, epochs=500):
    self._optim = torch.optim.Adam(self._model.parameters(), lr=lr)

    loss_values_x1 = []
    self._lambda = lam

    # training
    for _ in range(epochs):
      f = self._model(inputs.float())
      regularizer = 0
      if len(self._params) != 0:
        loss_ewc = 0
        for n, p in self._model.named_parameters():
          loss_ewc += torch.matmul(self._fim[n].T, (torch.reshape(p, (-1,1)) - torch.reshape(self._params[n], (-1,1))) ** 2)
        regularizer += self._lambda * loss_ewc

      loss = self._loss(f, labels.unsqueeze(1).float()) + regularizer
      self._optim.zero_grad()
      loss.backward()
      self._optim.step()

      # store loss
      loss_values_x1.append(loss.item())

    for n, p in deepcopy(self._model).named_parameters():
      if p.requires_grad:
        self._params[n] = p

    # update fisher information matrix
    f = self._model(inputs.float())
    loss = self._loss(f, labels.unsqueeze(1).float())
    self._optim.zero_grad()
    loss.backward()

    temp_fisher = {}
    for n, p in self._model.named_parameters():
      temp_fisher[n] = torch.reshape(p.grad.data, (-1,1))

    for n, p in temp_fisher.items():
      if n in self._fim:
        self._fim[n] = self._fim[n]*alpha + p**2 * (1-alpha)
      else:
        self._fim[n] = p**2
    return loss_values_x1
~~~

With the 
