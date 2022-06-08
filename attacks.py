"""
This file contains the implementations for the adversarial attacks:
    - Fast Gradient Sign Method
    - Projected Gradient Descent
    - DeepFool
    - Carlini & Wagner attack.
"""

import numpy as np
import torch

from tqdm import tqdm


class FGSM_Adversary:
    """ Implementation of the Fast Gradient Sign Method"""
    def __init__(self, epsilon):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbation (in 2-norm)
        """
        self.epsilon = epsilon
        self.bce_loss = torch.nn.BCELoss(reduction='sum')  # we maximize the loss w.r.t the predicted class rather than
                                                           # maximizing the corresponding logit, usually works best

    def __call__(self, model, x, y=None):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    y: class predicted by the classifier (which we want to change), torch.Tensor with shape (N, )

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        x = torch.Tensor(x)
        y = torch.Tensor(model.predict(x)) if y is None else torch.Tensor(y)
        x_adv = torch.autograd.Variable(torch.clone(x), requires_grad=True)

        # Get the gradient with respect to the loss
        loss_x = self.bce_loss(model(x_adv), y)
        loss_x.backward()
        grad = x_adv.grad

        # Do the optimal step wrt the l2 norm (epsilon-large)
        delta = grad / (torch.linalg.norm(grad, dim=-1, keepdims=True) + 1e-16) * self.epsilon
        x_adv = (x_adv + delta).detach()

        # Check if valid adversarial example
        valid_adversarial = model.predict(x_adv) != y.numpy()
        norm = torch.sqrt(torch.sum(delta**2, -1))

        return x_adv.numpy(), valid_adversarial, norm.numpy()

class PGD_Adversary:
    """ Implementation of the Projected Gradient Descent attack """
    def __init__(self, epsilon, lr=0.1, restarts=1, maxiters=1000, noise_std=0.01, scm=None, verbose_outer=False,
                 verbose_inner=False):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbation (in 2-norm)
                    lr: float, learning rate of the SGD optimizer
                    restarts: int, number of restarts used
                    maxiters: int, number of iterations for the optimization
                    noise_std: float, standard deviation of the noise added to x for each restart
                    scm: nn.Model, takes as input some intervention and returns the corresponding counterfactual
                    verbose_outer: bool, tqdm over the number of random restarts
                    verbose_inner: bool, tqdm over the optimization iterations (for each random restart)
        """
        self.epsilon = epsilon
        self.lr = lr  # 0.01 performs 1% better than 0.1 but slower (generally not worth it)
        self.restarts = restarts  # 20 performs 1% better than 1 but much slower (generally not worth it)
        self.rand_init = self.restarts > 1  # only use random initialization if there are multiple restarts
        self.maxiters = maxiters
        self.noise_std = noise_std
        self.verbose_inner = verbose_inner
        self.verbose_outer = verbose_outer
        self.scm = scm

    def __call__(self, model, x, y=None):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    y: class predicted by the classifier (which we want to change), torch.Tensor with shape (N, )

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        x = torch.Tensor(x)
        y = torch.Tensor(model.predict(x)) if y is None else torch.Tensor(y)
        target_vec = 1.0 - y # binary classifier, so only two labels

        # To store solutions for multiple random starts
        all_x_adv = np.copy(x)
        best_l2 = np.ones(x.shape[0]) * np.inf

        pbar = tqdm(range(self.restarts)) if self.verbose_outer else range(self.restarts)
        for i in pbar:
            if self.verbose_outer:
                pbar.set_description("left: %.2f" % (np.sum(best_l2 < np.inf) / x.shape[0]))
            x_adv, valid, l2 = self._optimize(model, x, target_vec)

            # Save best so far
            better = l2 < best_l2
            replace = np.logical_and(valid, better)
            best_l2[replace] = l2[replace]
            all_x_adv[replace] = x_adv[replace]

        valid_adversarial = best_l2 < np.inf  # solution found?

        return all_x_adv, valid_adversarial, best_l2

    def _optimize(self, model, x, target_vec):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    y: target class of the adversarial example, torch.Tensor with shape (N, )

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        unfinished = torch.ones(x.shape[0])

        noise = torch.normal(0, self.noise_std, x.shape) if self.rand_init else 0.
        delta = torch.autograd.Variable(torch.zeros(x.shape) + noise, requires_grad=True)

        optimizer = torch.optim.SGD([delta], self.lr)
        bce_loss = torch.nn.BCELoss(reduction='none')

        pbar = tqdm(range(self.maxiters)) if self.verbose_inner else range(self.maxiters)
        for step in pbar:
            if self.verbose_inner:
                pbar.set_description("left: %.2f" % (unfinished.sum() / x.shape[0]))

            optimizer.zero_grad()

            # Compute loss
            pertb = x + delta if self.scm is None else self.scm(delta) # IMF?
            loss = bce_loss(model(pertb), target_vec) # differentiate through both the SCM and the classifier

            # Apply mask over the ones where adversarial has already been found
            loss_mask = unfinished * loss
            loss_sum = torch.sum(loss_mask)

            # Update delta
            loss_sum.backward()
            optimizer.step()

            with torch.no_grad():
                # Project to L2 ball
                norm = torch.sqrt(torch.sum(delta**2, -1))
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

                # Found adversarial?
                unfinished = model.predict_torch(x + delta) != target_vec

            if not torch.any(unfinished):
                break

        x_adv = x + delta if self.scm is None else self.scm(delta)
        valid_adversarial = torch.logical_not(unfinished)
        norm = torch.sqrt(torch.sum(delta**2, -1))
        return x_adv.detach().cpu().numpy(), valid_adversarial.numpy(), norm.detach().cpu().numpy()


class DeepFool_Adversary:
    """ Implementation of the DeepFool attack """
    def __init__(self, restarts=1, maxiters=1000, noise_std=0.1, scm=None, verbose_outer=False, verbose_inner=False):
        """
        Inputs:     restarts: int, number of restarts used
                    maxiters: int, number of iterations for the optimization
                    noise_std: float, standard deviation of the noise added to x for each restart
                    scm: nn.Model, takes as input some intervention and returns the corresponding counterfactual
                    verbose_outer: bool, tqdm over the number of random restarts
                    verbose_inner: bool, tqdm over the optimization iterations (for each random restart)
        """
        self.restarts = restarts  # 20 performs 1% than 1 but it is much slower (not worth it)
        self.rand_init = self.restarts > 1
        self.maxiters = maxiters
        self.noise_std = noise_std
        self.verbose_inner = verbose_inner
        self.verbose_outer = verbose_outer
        self.scm = scm

    def __call__(self, model, x, y=None, subspace_mask=None):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    y: class predicted by the classifier (which we want to change), torch.Tensor with shape (N, )
                    subspace_mask: torch.Tensor or None, shape (1, D) with 0s or 1s, applied to the delta perturbation

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        x = torch.Tensor(x)
        y = torch.Tensor(model.predict(x)) if y is None else torch.Tensor(y)
        subspace_mask = subspace_mask if subspace_mask is not None else torch.ones(1, x.shape[-1])
        target_vec = 1.0 - y

        # To store solutions for multiple random starts
        all_x_adv = np.copy(x)
        best_l2 = np.ones(x.shape[0]) * np.inf

        pbar = tqdm(range(self.restarts)) if self.verbose_outer else range(self.restarts)
        for i in pbar:
            if self.verbose_outer:
                pbar.set_description("left: %.2f" % (np.sum(best_l2 < np.inf) / x.shape[0]))
            x_adv, valid, l2 = self._optimize(model, x, target_vec, i, subspace_mask)

            # Save best so far
            better = l2 < best_l2
            replace = np.logical_and(valid, better)
            best_l2[replace] = l2[replace]
            all_x_adv[replace] = x_adv[replace]

        return all_x_adv, best_l2 < np.inf, best_l2

    def _optimize(self, model, x, y_target, id, subspace_mask):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    y: target class of the adversarial example, torch.Tensor with shape (N, )
                    subspace_mask: torch.Tensor or None, shape (1, D) with 0s or 1s, applied to the delta perturbation

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        unfinished = torch.ones(x.shape[0])

        x = torch.Tensor(x)
        noise = torch.normal(0, self.noise_std, x.shape) * subspace_mask if self.rand_init and id > 0 else 0.
        delta = torch.autograd.Variable(torch.zeros(x.shape) + noise, requires_grad=True)
        optimizer = torch.optim.Adam([delta], 0, amsgrad=True) # only to get gradients

        pbar = tqdm(range(self.maxiters)) if self.verbose_inner else range(self.maxiters)
        for _ in pbar:
            if self.verbose_inner:
                pbar.set_description("left: %.2f" % (unfinished.sum() / x.shape[0]))

            optimizer.zero_grad()

            # Compute the logits f(x) (before the sigmoid activation)
            x_adv = x + delta if self.scm is None else self.scm(delta)
            fx = model.logits(x_adv)

            # Compute gradient of f w.r.t x
            sum_fx = torch.sum(fx)
            sum_fx.backward()
            grad_f = delta.grad * subspace_mask

            # Compute update direction
            norm_grad_f = torch.sum(grad_f ** 2, -1)
            ri = (unfinished * fx / (norm_grad_f + 1e-16))[:, None] * grad_f

            # Perform the update
            with torch.no_grad():
                delta[:] = delta - ri

            # Found adversarial?
            with torch.no_grad():
                unfinished = model.predict_torch(x_adv) != y_target

            if not torch.any(unfinished):
                break

        norm = torch.linalg.norm(delta, dim=-1)
        valid_adversarial = torch.logical_not(unfinished)
        return x_adv.detach().cpu().numpy(), valid_adversarial.numpy(), norm.detach().cpu().numpy()


class CW_Adversary:
    "Implementation of the Carlini & Wagner attack, adapted from https://github.com/kkew3/pytorch-cw2"
    def __init__(self, confidence=0.0, c_range=(1e-3, 1e10), search_steps=15, max_steps=1000, abort_early=True,
                 optimizer_lr=1e-2, init_rand=False, scmm=None, verbose=False):
        """
        Inputs:     confidence: float, confidence constant, kappa in the paper (set to 0 in the paper)
                    c_range: [float, float], range of c parameter in the loss, l2_loss + c * cw_loss
                    search_steps: int, the number of steps to perform binary search of c over c_range
                    max_steps: int, maximum number of inner optimization steps
                    abort_early: bool, early stopping inner optimization when loss stops increasing
                    optimizer_lr: float, learning rate of the optimizer
                    init_rand: bool, initialize the perturbation with small Gaussian noise
                    scmm: nn.Model, takes as input some intervention and returns the corresponding counterfactual
                    verbose: bool, print the progress of the optimizer (binary search steps)
        """
        self.confidence = float(confidence)
        self.c_range = (float(c_range[0]), float(c_range[1]))
        self.binary_search_steps = search_steps
        self.max_steps = max_steps
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.optimizer_lr = optimizer_lr
        self.init_rand = init_rand
        self.repeat = (self.binary_search_steps >= 10)  # largest c is attempted at least once, ensure some adversarial
                                                        # is found, even if with poor L2 distance
        self.constrains = None
        self.verbose = verbose
        self.scmm = scmm

    def __call__(self, model, x, interv, interv_set=None):
        """
        Inputs:     model: type trainers.Classifier
                    x: inputs to attack (factual), torch.Tensor with shape (N, D)
                    interv: recourse intervention, torch.Tensor with shape (N, D)
                    interv_set: set of features which are intervened upon, list of list of int, length N

        Outputs:    x_adv: adversarial example found, np.array with shape (N, D)
                    valid_adversarial: whether x_adv is assigned a different label than x by model, np.array shape (N, )
                    norm: 2-norm of (x_adv - x), np.array with shape (N, )
        """
        self.confidence = model.get_threshold_logits() - 1e-9  # since g(x) >= b rather than g(x) > b
        x = torch.Tensor(x)
        y = torch.ones(x.shape[0])

        batch_size = x.shape[0]
        y_np = y.clone().cpu().numpy()  # for binary search

        # Bounds for binary search
        c = np.ones(batch_size) * self.c_range[0]
        c_lower = np.zeros(batch_size)
        c_upper = np.ones(batch_size) * self.c_range[1]

        # To store the best adversarial examples found so far
        x_adv = x.clone().cpu().numpy()  # adversarial examples
        o_best_l2 = np.ones(batch_size) * np.inf  # L2 distance to the adversary

        # Perturbation variable to optimize. maybe this should be reset at each step (delta, optimizer?)
        init_noise = torch.normal(0, 1e-3, x.shape) if self.init_rand else 0.
        delta = torch.autograd.Variable(torch.zeros(x.shape) + init_noise, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.optimizer_lr)

        range_list = tqdm(range(self.binary_search_steps)) if self.verbose else range(self.binary_search_steps)
        # Binary seach steps
        for sstep in range_list:

            # If the last step, then go directly to the maximum c
            if self.repeat and sstep == self.binary_search_steps - 1:
                c = c_upper

            # Best solutions for the inner optimization
            best_l2 = np.ones(batch_size) * np.inf
            prev_batch_loss = np.inf  # for early stopping

            steps_range = tqdm(range(self.max_steps)) if self.verbose else range(self.max_steps)
            for optim_step in steps_range:

                batch_loss, l2, yps, advs = self._optimize(model, optimizer, x, interv, interv_set, delta, y, torch.Tensor(c))

                # Constrains on delta if relevant
                if self.constrains is not None:
                    with torch.no_grad():
                        # Satisfy the constraints on the features
                        delta[:] = torch.min(torch.max(delta, self.constrains[0]), self.constrains[1])

                # Early stopping (every 10 steps check if loss has increased sufficiently)
                if self.abort_early and optim_step % (self.max_steps // 10) == 0:
                    if batch_loss > prev_batch_loss * (1 - self.ae_tol):
                        break
                    prev_batch_loss = batch_loss

                # Update best attack found during optimization
                for i in range(batch_size):
                    if yps[i] == (1 - y_np[i]): # valid adversarial example
                        if l2[i] < best_l2[i]:
                            best_l2[i] = l2[i]
                        if l2[i] < o_best_l2[i]:
                            o_best_l2[i] = l2[i]
                            x_adv[i] = advs[i]

            # Binary search of c
            for i in range(batch_size):
                if best_l2[i] < np.inf:  # found an adversarial example, lower c by halving it
                    if c[i] < c_upper[i]:
                        c_upper[i] = c[i]
                    if c_upper[i] < self.c_range[1] * 0.1: # a solution has been found sufficiently early
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                else:
                    if c[i] > c_lower[i]:
                        c_lower[i] = c[i]
                    if c_upper[i] < self.c_range[1] * 0.1:
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                    else: # one order of magnitude more if no solution has been found yet
                        c[i] *= 10

        valid_adversarial = o_best_l2 < np.inf
        norm = np.sqrt(o_best_l2)
        return x_adv, valid_adversarial, norm

    def _optimize(self, model, optimizer, x, interv, interv_set, delta, y, c):
        """
        Inputs:     model: type trainers.Classifier
                    optimizer: torch.optim.Optimizer, with parameters delta
                    x: inputs to attack, torch.Tensor with shape (N, D)
                    interv: recourse intervention, torch.Tensor with shape (N, D)
                    interv_set: set of features which are intervened upon, list of list of int, length N
                    delta: current perturbation, torch.Variable with shape (N, D)
                    y: class predicted by the classifier (which we want to change), torch.Tensor with shape (N, )
                    c: float, weight given to the C&W loss

        Outputs:    loss: float, overall loss (L2 + C&W)
                    norm: 2-norm of delta, np.array with shape (N, )
                    yp: classifier prediction for x_adv, np.array with shape (N, )
                    x_adv: adversarial example found, np.array with shape (N, D)
        """
        # Causal perturbation model
        D = x.shape[1]
        if self.scmm is None:
            x_adv = x + interv + delta
        else:
            x_prime = self.scmm.counterfactual(x, delta, np.arange(D), [True] * D)
            x_adv = self.scmm.counterfactual_batch(x_prime, interv, interv_set)  # counterfactual

        # Compute logits of adversary
        z, yp = model.logits_predict(x_adv)

        # Second term of loss (C&W loss): max(z, -k) for y=0, max(-z, -k) for y=1 (for binary classification)
        mask = 2 * y - 1
        loss_cw = torch.clamp(mask * z - self.confidence, min=0.0)

        # First term of loss: l2 norm of the perturbation
        l2_norm = torch.sum(torch.pow(delta, 2), -1)

        # Overall loss with parameter c
        loss = torch.sum(l2_norm + c * loss_cw)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item(), l2_norm.detach().cpu().numpy(), yp.detach().cpu().numpy(), x_adv.detach().cpu().numpy()
