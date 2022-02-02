---
layout: post
title: "EWC"
description: Introduction to EWC
---



Motivation for Elastic Weight Consolidation (EWC)
============
Although L-2 norm regularization moderates catastrophic forgetting in some sense, it has one severe problem: no distinction in feature importance
of previous tasks. As a result, L2-norm regularization may pose severe restrictions for all features, and, overall, the restriction can be so severe that the
neural network can only remember previous tasks at the expense of not learning the new task. In order to address this problem,
elastic weight consolidation (EWC) comes to the rescue. EWC is able to distinguish between important and unimportant features, and will
penalize features that are critical to previous tasks severely while penalizing marginal features slightly. This allows simultaneous remembering and learning



Idea behind EWC
============

EWC tackles the problem from a probabilistic perspective. Assume that we are trying to continually learn from a collection of datasets, D. The
conditional probability that we are trying to optimize would be *<span>log p(θ | D)</span>*. Let's first consider the two-task case.

Suppose *<span>D</span>* is comprised of independent and disjoint datasets *<span>D<sub>A</sub></span>* and
*<span>D<sub>B</sub></span>*, and it follows that *<span>D = D<sub>A</sub> ∪ D<sub>B</sub></span>*. For the 
two-task case, the conditional probability *<span>log p(θ|D)</span>* is equivalent to *<span>log p(θ|D<sub>A</sub> + D<sub>B</sub>)</span>*.
Using Beyes' rule, we can compute *<span>log p(θ | D)</span>* in the following way:

<p align="center">
    log p(D<sub>B</sub> | θ) + log p(θ | D<sub>A</sub>) - log p(D<sub>B</sub>)
</p>

*<span>log p(θ | D)</span>* is the posterior of continually learning two tasks, and terms in the above expression
corresponds to the negative loss of the second task, prior of the second task (also posterior of the first task),
and the normalization respectively. It can be easily inferred that all information about previous task should be contained
in the term *<span>log p(θ | D<sub>A</sub>)</span>*. Nevertheless, the exact posterior is intractable and 
we do not have access to data of previous tasks, so it must be approximated cleverly. One way to achieve this is through Laplace 
Approximation, which will be discussed briefly here.

The crux of Laplace approximation is Taylor expansion. Denote *<span> h(θ) = log p(θ | D<sub>A</sub>)</span>*, and let *<span>θ*</span>* be the point where *<span>h(θ)</span>*
is optimum. Second degree Taylor expansion would give us an approximation of *<span>h(θ)</span>*:

![Taylor_Expansion] (https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Taylor_expansion_eq3.jpg?raw=true)



$\frac{a}{b}$





How offline EWC works
============

Offline EWC is the naive version of EWC. It strictly follows the idea of EWC by storing all fisher information matrices from previous tasks,
and adding them one by one as the regularization term when learning a new task. Assume our model f has learnt T-1 tasks and
wants to learn the Tth one

<p align="center">
    min<sub>f</sub> L<sub>T</sub> = &alpha; F<sub>old</sub> + (1 - &alpha;) S
</p>



Implementation of offline EWC
============

The offline EWC is implemented below using pytorch

~~~python
class OfflineEWC:
    def __init__(self, model: nn.Module, loss=nn.MSELoss()):
        self._model = model

        self._params = []
        self._fims = []
        self._loss = loss
        self._optim = None
        # self._lambda = []

    def train(self, inputs, labels, lam, lr=8e8, epochs=500):

        self._optim = torch.optim.Adam(self._model.parameters(), lr=lr)

        loss_values_x1 = []

        # First training period
        for _ in range(epochs):

            f = self._model(inputs.float())

            regularizer = 0

            for n, p in self._model.named_parameters():
                for i in range(len(self._fims)):
                    regularizer += torch.dot(self._fims[i][n].reshape(-1), ((p - self._params[i][n]) ** 2).reshape(-1))

            loss = self._loss(f, labels.unsqueeze(1).float()) + lam * regularizer
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            # calculate and store the loss per epoch for both datasets
            loss_values_x1.append(loss.item())

        self._params.append({})
        temp_param = {n: p for n, p in self._model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(temp_param).items():
            self._params[-1][n] = p

        f = self._model(inputs.float())
        loss = self._loss(f, labels.unsqueeze(1).float())
        self._optim.zero_grad()
        loss.backward()

        temp_fisher = {}
        for n, p in self._model.named_parameters():
            temp_fisher[n] = p.grad.data

        self._fims.append({})
        for n, p in temp_fisher.items():
            self._fims[-1][n] = p ** 2

        return loss_values_x1
~~~

Demo of offline EWC
============

Next, we will try to convince you that offline EWC works through an example of four individual tasks. The data on which we're trying to train
continually is the following, and we will be using a 4-hidden-layer MLP with perceptron number of 1, 100, 100, 100, 100, and 1.

![offline4_data](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/data_online4.png?raw=true)


What can be improved?
============

The advantage of using offline EWC is obvious: it alleviates the problem of catastrophic forgetting and mimic the effect
of Hessian matrix to the greatest degree. However, its downside can also be annoying. Imagine a situation such that there are hundreds of thousands
tasks waiting to be learnt. Offline EWC will perform badly since it tries to store fisher information matrix for each task being
learnt, and there will be hundreds of thousands of them. So, in this case, not only the space consumption will be large, but also the
computation cost wil be huge.

Considering these two problems, online EWC has been introduced on the basis of offline EWC.
