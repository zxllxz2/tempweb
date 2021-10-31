---
layout: post
title: "Why you forget me?"
description: Explanation on catastrophic forgetting
---
<!-- Example modified from [here](http://www.unexpected-vortices.com/sw/rippledoc/quick-markdown-example.html){:target="_blank"}. -->

What is catastrophic forgtting
============

In real life, it is usual for a person to encounter a large number of tasks, and the person needs to learn those tasks one by one. As a common sense, after the person finishes learning the last task, it should still remember the all task learned before. However, does this common sense work for the Neural Network as well?


Numerical experiments
------------

Here is a simple example in 2 dimension. Let's form two disjoint datasets in dimension 2. Regaring them as two independent tasks, we will train our nerual network model on these two datasets sequentially. Both datasets are formed by polynomials with some noise. Below is the visualization for the data that we use, together with their original functions.
![example image](assets/images/datasets1.png)

We then define our Multi-Layer-Perceptron model. Here we are using PyTorch to conduct the experiment.

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

And we set the desired learning rate, number of epochs, loss function, and the optimizer.

~~~python
lr = 1e-2
n_epochs = 200
model = MLP(architecture=[1,150,150,1])
loss_f = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)
~~~

Great! Let's start training on our first task! While training, we plot the MSE loss on both datasets. We see clearly see that the loss for task 1 drops significantly. Since we do not train on the 2nd dataset, it is normal to have its loss grow.
![loss_after_task1_1](assets/images/loss_after_task1_1.png)

And we can visualize our regressor after training on the task 1 dataset.
![regressor_after_task1](assets/images/regressor_after_task1.png)

It looks nice, isn't it? Now we will train our model "continually" on the 2nd dataset. We plot the MSE loss on both datasets as well.
![loss_after_task2_1](assets/images/loss_after_task2_1.png)

The loss curve looks really weird. The loss for the 2nd dataset decreases, while the loss for the previously trained 1st dataset increases. Then, how about the final regressor we get?
![regressor_after_task2](assets/images/regressor_after_task2.png)

The regressor becomes a mess. Though it predicts data in task 2 accurately, it almost forgets task 1 completely. We can watch an animation of our training process to see such forgetting clearer.

<video width="320" height="240" autoplay>
  <source src="./assets/images/training1.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

Such a forgetting phenomenon appeared in Neural Network training is the so-called "catastrophic forgetting". While people learn a model for a large number of tasks sequentially where the data in the old tasks are not available any more during training new ones, catastrophic forgetting happens quite a few as model keeps forgetting knowledge obtained from the preceding tasks.


More experiments
------------

Still under construction. Coming soon!


How to solve catastrophic forgtting
============

Unfortunately, the catastrophic forgetting is still a problem in the continual learning area. The most basic way to deal with it would be that whenever we have a new task, we combine data from all tasks together and train on this overall dataset to obtain the model. However, the costs are expensive.

Some studies also try to alleviate catastrophic forgetting, utilizing architectural, functional, structural approaches. These include the method we are going to survey, like SWE, SI, and so on. They alter the architecture of the network, add a regularization term to the objective for penalization, or add penalties on network parameters to make them stay close to the parameters for the old tasks. We will go through them in ways that people can visualize and understand more easily.

Thanks for reading. If you like this article or are interested in the topic of catastrophic forgetting, you are more than welcome to explore our other projects. Thanks again for your support!
