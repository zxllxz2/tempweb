---
layout: post
title: "Why you forget me?"
description: Explanation on catastrophic forgetting
---


What is catastrophic forgtting
============

In real life, it is usual for a person to encounter a large number of tasks, and the person has to learn those tasks one by one. As common sense, the person would still remember almost all previously-learned tasks after finishing learning the last task. However, does this apply to the neural network as well? Although artificial neural networks can learn from different tasks, recognize patterns, and predict according to knowledge obtained, can the neural network master all knowledge as well as, or maybe better than, humans, when different tasks are given for training in a particular order? This article will tell you the answer.


Numerical experiments
------------

Here is a simple example in the 2D dimension where we have two disjoint datasets for the experiment. We will train our neural network to fit the first dataset, which is the first task, and then to fit the second dataset, which is the second task. Both datasets are generated from polynomials with some noise. Below is the visualization for the datasets, together with their original functions.

<img src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/datasets1.png?raw=true"><img/><br>
We defined our Multi-Layer-Perceptron mode using PyTorch.

~~~python
import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self,architecture=[1,100,100,1]):
    super(MLP, self).__init__()   

    self.architecture = architecture
    self.activation = nn.Sigmoid()

    # ading layers
    arch=[]
    for i in range(1, len(architecture) - 1):
        arch.append(nn.Linear(architecture[i - 1], architecture[i]))            
        arch.append(self.activation)

    self.basis = nn.Sequential(*arch)
    self.regressor = nn.Linear(architecture[-2], architecture[-1])

  def forward(self, f):
    assert f.shape[1] == self.architecture[0]
    z = self.basis(f)
    out = self.regressor(z)
    return out
~~~

And we set the desired learning rate, number of epochs, loss function, and optimizer.

~~~python
lr = 1e-2
n_epochs = 200
model = MLP(architecture=[1,150,150,1])
loss_f = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)
~~~

Great! Let's start our first task! While training, we plot the MSE loss on both datasets. We can see that the loss for task 1 drops significantly. Since we do not train on the second dataset, it is not surprising to see its loss grow.

![loss_after_task1_1](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss_after_task1_1.jpg?raw=true "loss after training on task 1")

And we can visualize our regressor after the first task.

![regressor_after_task1](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/regressor_after_task1.png?raw=true "regressor after training on task 1")

It looks Okay. Now we will continue to train our model on the second dataset. We plot the MSE loss on both datasets as well.

![loss_after_task2_1](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss_after_task2_1.png?raw=true "loss after training on task 2")

The loss curve looks really weird. The loss for the second dataset decreases, while the loss for the previously trained dataset increases dramatically. Then, how about the final regressor we get?

![regressor_after_task2](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/regressor_after_task2.png?raw=true "regressor after training on task 2")

The regressor becomes a mess. Although the model predicts data in task 2 accurately, it almost forgets everything learned from the first task. We can watch an animation of our training process to visualize this forgetting phenomenon.<br>
![loss_after_task1_1](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/training1.gif?raw=true "regressor training")<br>
Such a forgetting phenomenon that appeared in our training is the so-called **Catastrophic forgetting**. When people train a model on a large number of tasks sequentially, where the data of old tasks are not available anymore during training new ones, catastrophic forgetting always happens, as the model keeps forgetting knowledge obtained from the preceding tasks.


More experiments
------------
Besides disjoint datasets, here are two experiments on joint datasets generated by the same function.

These two datasets are generated by the radial basis function. We can see the animation of fitting a model on the two datasets sequentially.<br>
![RBF_forgetting](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/RBF_forgetting.gif?raw=true)<br>
This clearly shows the forgetting phenomenon. We can also check the MSE loss curves.

![RBE_loss](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/same_func_loss.jpg?raw=true)

Next, these two datasets are generated by the sigmoid function. We can see the animation of fitting a model on the two datasets sequentially.<br>
![Sigmoid_forgetting](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/sigmoid_forgetting.gif?raw=true)<br>
Also checks for the MSE loss curves.

![Sigmoid_loss](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/same_func_loss2.jpg?raw=true)

How to solve catastrophic forgtting
============

Now, we have seen some toy examples of catastrophic forgetting. However, is it a significant problem to notice? In fact, the catastrophic forgetting phenomenon is extremely important in the industrial setting, as the target function or the training dataset of a model is always subject to unpredictable changes from the market. Therefore, the loss of predicting ability on the previously trained data can be devastating.

Unfortunately, catastrophic forgetting is still an unsolved problem in the continual learning area. A simple solution, which is currently the most effective solution, is to ensure that data from all tasks can be simultaneously available during future training. In this case, for any future task, we combine its own data with data from previous tasks and optimize our model on this huge overall dataset. This approach, in fact, would yield an upper bound for the performance of any continual learning model. However, it usually requires a memory system to remember previous task data and replay them during training on a new task, which is impractical with a massive amount of tasks as the memory cost would be unaffordable.

<p align="center">
<img src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/annoyed.jpg?raw=true" alt="drawing" width="240"><img/>
</p>

Luckily, some studies invented other methods to alleviate catastrophic forgetting, which can be broadly divided into three main categories - architectural, regularization-based, and memory-based. All those methods reduce forgetting to some extent with limitations. What we are going to discuss here are regularization-based methods, including the most basic L2-norm regularization and the Elastic Weight Consolidation (EWC). Basically, regularization-based methods apply constraints on the model, forcing model parameters to stay close to optimized values for the old tasks. We will go through both of them in later sections.

Thanks for reading. If you like this article or are interested in the topic of catastrophic forgetting, you are more than welcome to read our other project posts. Thanks again for your support!
