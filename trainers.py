"""
This file contains the implementation for the Logistic Regression and MLP classifiers, as well as the methods to train
these classifiers using:
    - Expected Risk Minimization (ERM)
    - Only the actionable features (AF)
    - Actionable Local Linear Regularization (ALLR)
    - The regularizer of Ross et al. (ROSS)
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter

def mcc_scores(probs, Y_test, N=101):
    """
    Return the mcc score of the classifier as a function of the chosen decision threshold.

    Inputs:     probs: np.array (M,), the score associated with p(y=1|x)
                Y_test: np.array (M,), labels for each instance
                N: int, we evaluate N different decision thresholds in the range [0, 1]

    Returns:    thresholds: np.array (N,), decision thresholds for which mcc was evaluated
                mccscores: np.array (N,), corresponding mcc values
    """
    thresholds = np.linspace(0, 1, N)
    mccscores = np.zeros(N)
    for i in range(N):
        yp = (probs >= thresholds[i]).astype(np.int)
        mccscores[i] = matthews_corrcoef(Y_test, yp)
    return thresholds, mccscores


class Classifier(torch.nn.Module):
    """
    Classifier h(x) = sigmoid(g(x)) > b, where
        g(x) is self.forward (the logits of the classifier), and
        b is self.threshold
    """
    def __init__(self, actionable_mask=False, actionable_features=None, threshold=0.):
        """
        Inputs:     actionable_mask: If True, only actionable features are used as input to the classifier
                    actionable_features: Indexes of the actionable features (e.g. [0, 3]), used for regularization
                    threshold: float, b in h(x) = sigmoid(g(x)) > b
        """
        super(Classifier, self).__init__()

        self.actionable_mask = actionable_mask
        self.actionable_features = actionable_features
        self.threshold = threshold
        self.sigmoid = torch.nn.Sigmoid()

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_max_mcc_threshold(self, X, Y):
        """
        Sets the decision threshold as that which maximizes the mcc score.

        Inputs:     X: np.array or torch.Tensor (N, D), training data
                    Y: np.array (N, ), labels
        """
        probs = self.probs(torch.Tensor(X)).detach().numpy()
        thresholds, mccs = mcc_scores(probs, Y)
        max_indx = np.argmax(mccs)
        self.set_threshold(thresholds[max_indx])
        return mccs.max()

    def get_threshold(self):
        return self.threshold

    def get_threshold_logits(self):
        def logit(p):
            p = min(max(p, 0.001), 0.999) # prevent infinities
            return np.log(p) - np.log(1 - p)
        return logit(self.threshold)

    def logits(self, x):
        """
        Returns g(x)

        Inputs:   data samples x as torch.Tensor with shape (B, D)
        Returns:  logits of the classifier as a torch.Tensor with shape (B,)
        """
        if self.actionable_mask:
            x = x[..., self.actionable_features]
        return self.g(x).reshape(-1)

    def probs(self, x):
        """
        Returns p(y = 1 | x) = sigmoid(g(x))

        Inputs:   data samples x as torch.Tensor with shape (B, D)
        Returns:  p(y = 1 | x) as a torch.Tensor with shape (B,)
        """
        return self.sigmoid(self.logits(x))

    def predict_torch(self, x):
        """
        Inputs: data samples x as torch.Tensor with shape (B, D)
        Outputs: predicted labels as torch.Tensor of dtype int and shape (B,)
        """
        return (self.probs(x) >= self.threshold).to(torch.int)

    def predict(self, x):
        """
        Inputs: data samples x as torch.Tensor with shape (B, D)
        Outputs: predicted labels as np.array of dtype int and shape (B,)
        """
        return self.predict_torch(torch.Tensor(x)).cpu().detach().numpy()

    def logits_predict(self, x):
        logits = self.logits(x)
        return logits, (self.sigmoid(logits) >= self.threshold).to(torch.int)

    def probs_predict(self, x):
        logits, predict = self.logits_predict(x)
        return self.sigmoid(logits), predict

    def forward(self, x):
        return self.logits(x)


class LogisticRegression(Classifier):
    """
    Implementation of a linear classifier, where g(x) = <w, x> + b
    To be trained using logistic regression, that is, p(y=1|x) = sigmoid( g(x) )
    """
    def __init__(self, input_dim, allr_reg=False, **kwargs):
        """
        Inputs:    input_dim: Number of features of the data samples
                   allr_reg: if True, L2 regularization of the actionable features
        """
        super().__init__(**kwargs)
        self.allr_reg = allr_reg
        self.og_input_dim = input_dim
        actual_input_dim = len(self.actionable_features) if self.actionable_mask else input_dim
        self.g = torch.nn.Linear(actual_input_dim, 1)

        if allr_reg and not self.actionable_mask:
            self.unactionable_mask = torch.ones(self.g.weight.shape)
            self.unactionable_mask[0, self.actionable_features] = 0.

    def get_weight(self):
        """ Returns: weights of the linear layer as a torch.Tensor with shape (1, self.og_input_dim) """
        if self.actionable_mask: # If using the mask all other features have a weight of 0
            weight = torch.zeros((1, self.og_input_dim))
            weight[0, self.actionable_features] = self.g.weight.reshape(-1)
            return weight
        else:
            return self.g.weight

    def get_weights(self):
        """
        For the classifier h(x) = <w,x> > b

        Returns:    w as an np.array of shape (self.og_input_dim, 1)
                    b as an np.array of shape (1, )
        """
        def logit(p):
            p = min(max(p, 0.001), 0.999) # prevent infinities
            return np.log(p) - np.log(1 - p)
        w = self.get_weight().cpu().detach().numpy().T
        b = logit(self.threshold) - self.g.bias.cpu().detach().numpy()
        return w, b

    def regularizer(self):
        """
        Returns the relevant regularization quantity, as determined by self.allr_reg

        Returns: torch.Tensor of shape (,)
        """
        if not self.allr_reg or (self.allr_reg and self.actionable_mask):
            return 0.
        return torch.sum((self.g.weight * self.unactionable_mask) ** 2)


class MLP(Classifier):
    """ Implementation an MLP classifier, where p(y=1|x) = sigmoid( g(x) ) and g(x) is a 3-layer MLP."""
    def __init__(self, input_dim, hidden_size=100, **kwargs):
        """
        Inputs:  input_dim: Number of features of the data samples
                 hidden_size: Number of neurons of the hidden layers
        """
        super().__init__(**kwargs)

        input_dim = len(self.actionable_features) if self.actionable_mask else input_dim
        self.g = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_size),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_size, hidden_size),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_size, 1))

    def regularizer(self): # For the MLP classifier regularization is done by using different Trainers
        return 0.


class Trainer:
    def __init__(self, lr=0.001, batch_size=100, lambda_reg=0, pos_weight=None, device='cpu', verbose=False,
                 print_freq=1, tb_folder=None, save_freq=None, save_dir=None):
        """
        Base trainer class implementing gradient descent (with Adam as the optimizer).

        Inputs:  batch_size: int
                 lr: float, learning rate
                 print_freq: int, frequency at which certain metrics are reported during training
                 lambda_red: float, regularization strength
                 verbose: bool, whether to print certain metrics (accuracy, f1, auc) during training
                 device: 'cpu' or 'cuda'
                 pos_weight: argument for torch.nn.BCEWithLogitsLoss, used for unbalanced data sets
                 tb_folder: None or str, if not None the folder where to save TensorBoard data
                 save_freq: int, model is saved after save_freq number of epochs
                 save_dir: str, location where the model is saved
        """
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = torch.device(device)
        self.verbose = verbose
        self.print_freq = print_freq
        pos_weight = torch.Tensor([pos_weight]) if pos_weight is not None else None
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        self.tb_writer = SummaryWriter(tb_folder) if tb_folder is not None else None
        self.save_model = (save_freq is not None) and (save_dir is not None)
        self.save_freq = save_freq
        self.save_dir = save_dir

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs):
        """
        Does gradient descent over a number of epochs.

        Inputs:  model: type Classifier
                 X_train, Y_train: training data, np.array or torch.Tensor, shape (N, D) and (N, ) respectively
                 X_test, Y_test: testing data, np.array or torch.Tensor, shape (M, D) and (M, ) respectively
                 epochs: int, number of training epochs
        """
        def performance_metrics(model, X_train, Y_train, X_test, Y_test):
            prev_threshold = model.get_threshold()
            model.set_max_mcc_threshold(X_train, Y_train)
            mcc = matthews_corrcoef(model.predict(X_test), Y_test)
            acc = (model.predict(X_test) == Y_test).sum() / X_test.shape[0]
            model.set_threshold(prev_threshold)
            return float(acc), float(mcc)

        X_test, Y_test = torch.Tensor(X_test).to(self.device), torch.Tensor(Y_test).to(self.device)
        X_train, Y_train = torch.Tensor(X_train).to(self.device), torch.Tensor(Y_train).to(self.device)

        train_dst = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        accuracies = np.zeros(int(epochs / self.print_freq))

        prev_loss = np.inf
        val_loss = 0
        for epoch in range(epochs):
            for x, y in train_loader:
                optimizer.zero_grad()

                loss = self.get_loss(optimizer, model, x, y)
                loss += self.lambda_reg * model.regularizer()

                loss.backward()
                optimizer.step()

            if (self.verbose or (self.tb_writer is not None)) and (epoch % self.print_freq == 0):
                mean_acc, mcc_max = performance_metrics(model, X_train, Y_train.numpy(), X_test, Y_test.numpy())

                if self.verbose:
                    print("E: %d Acc: %.4f mcc: %.4f" % (epoch, mean_acc, mcc_max))

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('acc', mean_acc, epoch)
                    self.tb_writer.add_scalar('mcc', mcc_max, epoch)

            if self.save_model:
                if (epoch % self.save_freq) == 0 and epoch > 0:
                    torch.save(model.state_dict(), self.save_dir + '_e' + str(epoch) + '.pth')

        if self.save_model:
            torch.save(model.state_dict(), self.save_dir + '.pth')

        # Return the model evaluation
        return performance_metrics(model, X_train, Y_train.numpy(), X_test, Y_test.numpy())



class ERM_Trainer(Trainer):
    """ Expect Risk Minimization (just use the loss without any regularization) """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        return self.loss_function(model(x), y)


class Adversarial_Trainer(Trainer):
    """ Min Max, max for a ball around x """
    def __init__(self, epsilon, alpha=0.1, adversarial_steps=7, actionable_dirs=None, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbations
                    alpha: learning rate for PGD maximization
                    adversarial_steps: numer of steps for PGD maximization
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.adversarial_steps = adversarial_steps
        self.actionable_dirs = 1. if actionable_dirs is None else actionable_dirs

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        x_adv = self.get_adversarial(model, x, y)
        optimizer.zero_grad()
        return self.loss_function(model(x_adv), y)

    def get_adversarial(self, model, x, y):
        """
        Returns argmax_{x_adv \in ||x_adv - x|| <= epsilon} loss(h(x_adv), y)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        if self.adversarial_steps == 1:
            return self.fgsm_step(model, x, y)
        else:
            return self.pgd_step(model, x, y, self.adversarial_steps)

    def fgsm_step(self, model, x, y):
        """
        Obtain x_adv using the Fast Gradient Sign Method

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        x.requires_grad_(True)
        loss_x = self.loss_function(model(x), y)
        grad = torch.autograd.grad(loss_x, x)[0]
        x.requires_grad_(False)

        delta = grad / (torch.linalg.norm(grad, dim=-1, keepdims=True) + 1e-16) * epsilon
        return (x + delta).detach()

    def pgd_step(self, model, x, y, n_steps):
        """
        Obtain x_adv using Projected Gradient Ascent over the loss

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    n_steps: int, number of optimization steps

        Returns:    torch.Tensor, shape (B, D)
        """
        x_adv = torch.autograd.Variable(torch.clone(x), requires_grad=True)
        optimizer = torch.optim.Adam([x_adv], self.alpha)

        for step in range(n_steps):
            optimizer.zero_grad()

            loss_x = -self.loss_function(model(x_adv), y)
            loss_x.backward()
            optimizer.step()

            # Project to L2 ball
            with torch.no_grad():
                delta = (x_adv - x) * self.actionable_dirs
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / (norm[too_large, None] + 1e-8) * self.epsilon
                x_adv[:] = x + delta

        return x_adv.detach()


class TRADES_Trainer(Trainer):
    """
    Original paper: https://arxiv.org/pdf/1901.08573.pdf
    Code aapted from https://github.com/yaodongyu/TRADES
    """
    def __init__(self, epsilon, adversarial_steps=10, beta=1., mask=None, **kwargs):
        """
        Input:  epsilon: float, maximum magnitude of the adversarial perturbations
                adversarial_steps: int, number of steps when searching for the adversarial perturbation
                beta: float, beta parameter in TRADES
                mask: None for no mask, otherwise torch.Tensors with 0's or 1's (mask applied to perturbation)
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.beta = beta
        self.mask = mask
        self.trades_loss = torch.nn.KLDivLoss(size_average=False)

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        x_adv = self.get_adversarial(model, x, y)
        optimizer.zero_grad()
        return self.loss_function(model(x), y) + self.beta * self.trades_loss(model.probs(x_adv), model.probs(x))

    def get_adversarial(self, model, x, y):
        """
        Obtain x_adv using Projected Gradient Ascent over the TRADES loss

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        noise = 0.001 * torch.randn(x.shape)
        delta = torch.autograd.Variable(torch.zeros(x.shape) + noise, requires_grad=True)
        optimizer = torch.optim.Adam([delta], self.epsilon / self.adversarial_steps * 2)

        for step in range(self.adversarial_steps):
            optimizer.zero_grad()

            # Get gradient
            loss_x = -self.trades_loss(torch.log(model.probs(x + delta)), model.probs(x))
            loss_x.backward()

            optimizer.step()

            # Project to L2 ball
            with torch.no_grad():
                if self.mask is not None:
                    delta[:] = delta * self.mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

        return (x + delta).detach()


class LLR_Trainer(Trainer):
    """ Local Linear Regularizer, as described in https://arxiv.org/pdf/1907.02610.pdf, Algorithm 1, Appendix E """
    def __init__(self, epsilon, adversarial_steps=10, step_size=0.1, lambd=4., mu=3., use_abs=False,
                 reg_loss=False, grad_penalty=2, linearity_mask=None, gradient_mask=None, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbations
                    adversarial_steps: int, number of steps when searching for the adversarial perturbation
                    step_size: float, learning rate of the optimizer when searching for the adversarial perturbation
                    lambd: float, lambda parameter in LLR, corresponding to linearity
                    mu: float, mu parameter in LLR, corresponding to the magnitude of the gradient
                    use_abs: if True, uses the absolute value of g (Equation 5 in the LLR paper), not in their code
                    reg_loss: if True, regularizers the loss as in the LLR paper, if False regularizes the logits
                    grad_penalty: norm used to penalize the magnitude of the gradient (0 for inner product, 1 for
                                  l1 norm, 2 for l2 norm)
                    linearity_mask: None for no mask, otherwise torch.Tensors with 0's or 1's (mask applied to when
                                    searching for the adversarial violation of the linearity constraint)
                    gradient_mask: mask applied to gradient when penalizing the magnitude of the gradient
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.step_size = step_size
        self.lambd = lambd
        self.mu = mu
        self.use_abs = use_abs
        self.reg_loss = reg_loss
        self.grad_penalty = grad_penalty
        self.linearity_mask = None if linearity_mask is None else torch.Tensor(linearity_mask).reshape(1, -1).to(self.device)
        self.gradient_mask = 1. if gradient_mask is None else torch.Tensor(gradient_mask).reshape(1, -1).to(self.device)


    def grad_fx(self, model, x, y, grad_model=False):
        """
        Calculates either l(h(x), y) or g(x), and its loss w.r.t x

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    grad_model: bool, True if we want to differentiate the gradient w.r.t the model parameters

        Returns:    loss_x: torch.Tensor with shape (,), the mean (resp. sum) of loss(h(x), y) (resp. g(x))
                            depending on self.reg_loss (whether we regularize the loss as in LLR or g(x))
                    grad_loss_x: torch.Tensor with shape (B, D), gradient of loss_x w.r.t x
        """
        x.requires_grad_(True)
        loss_x = self.loss_function(model(x), y) if self.reg_loss else model.logits(x)
        grad_loss_x = torch.autograd.grad(torch.sum(loss_x), x, create_graph=grad_model)[0]
        x.requires_grad_(False)

        if not grad_model:
            loss_x = loss_x.detach()
        return loss_x, grad_loss_x

    def g(self, model, x, y, delta, grad, loss_x):
        """
        Evaluates the local linearity measure (LLR paper Equation 5)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D)
                    grad: gradient of loss(h(x), y) or g(x) w.r.t x
                    loss_x: loss(h(x), y) or g(x)

        Returns: g(delta, x), shape (B, D)
        """
        loss_pertb = self.loss_function(model(x + delta), y) if self.reg_loss else model.logits(x + delta)
        g_term = loss_pertb - loss_x - torch.sum(delta * grad, -1)
        if self.use_abs:
            g_term = torch.abs(g_term)
        return g_term

    def get_perturb(self, model, x, y):
        """
        Optimize for the perturbation delta which maximizes the  local linearity measure g(delta, x)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    delta: torch.Tensor, shape (B, D)
        """
        loss_x, grad = self.grad_fx(model, x, y)

        noise = self.epsilon * torch.randn(x.shape)
        delta = torch.autograd.Variable(noise.to(self.device), requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for _ in range(self.adversarial_steps):
            optimizer.zero_grad()
            loss = -torch.mean(self.g(model, x, y, delta, grad, loss_x=loss_x))
            loss.backward()
            optimizer.step()

            # Project to L2 ball, and with the linearity mask
            with torch.no_grad():
                if self.linearity_mask is not None:
                    delta[:] = delta * self.linearity_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

        return delta.detach()

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        # Due to the absolute value, the loss must not be reduced
        self.loss_function.reduction = 'none'

        # Calculate delta perturbation using projected gradient ascent
        delta = self.get_perturb(model, x, y)

        optimizer.zero_grad()
        loss_x, grad_loss_x = self.grad_fx(model, x, y, grad_model=True)

        loss2 = self.g(model, x, y, delta, grad_loss_x, loss_x)  # local linearity measure
        if not self.reg_loss:
            loss_x = self.loss_function(model(x), y)  # normal loss as in ERM

        if self.grad_penalty == 2:
            loss3 = torch.sum((grad_loss_x * self.gradient_mask) ** 2, -1)
        if self.grad_penalty == 0:
            loss3 = torch.abs(torch.sum(delta * grad_loss_x * self.gradient_mask , -1))
        if self.grad_penalty == 1:
            loss3 = torch.sum(torch.abs(grad_loss_x * self.gradient_mask), -1)

        return torch.mean(loss_x + self.lambd * loss2 + self.mu * loss3)


class Ross_Trainer(Trainer):
    """
    Regularizer proposed by Ross et al. in ''Learning Models for Actionable Recourse''

    Rather than considering min max as in adversarial training, they propose to regularize with min min.
    The perturbations over the inner minimization are projectd to the actionable set.
    """
    def __init__(self, epsilon, lambd, actionable_mask, epsilon_AT=0.1, AT=False, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the perturbations
                    lambd: float, regularization weight
                    actionable_mask: list, actionable_mask[i] = 1 --> feature i is actionable
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.epsilon_AT = epsilon_AT
        self.lambd = lambd
        self.AT = AT
        self.actionable_mask = torch.Tensor(actionable_mask).reshape(1, -1)
        self.unactionable_mask = 1.0 - self.actionable_mask

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        if self.AT:
            x_adv = self.pgd_step(model, x, y, 7)
        x_ross = self.fgsm_step(model, x)

        yp = torch.ones(x.shape[0])
        optimizer.zero_grad()
        loss1 = self.loss_function(model(x_adv), y) if self.AT else self.loss_function(model(x), y)
        loss2 = self.loss_function(model(x_ross), yp)
        loss = loss1 + self.lambd * loss2
        return loss

    def fgsm_step(self, model, x):
        """
        Calculates x_adv = min_{\delta \in actionable} l(g(x + delta), 1)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        # Gradient of loss w.r.t all instances favourably classified
        x.requires_grad_(True)
        yp = torch.ones(x.shape[0])
        loss_x = self.loss_function(model(x), yp)
        grad = torch.autograd.grad(loss_x, x)[0]
        x.requires_grad_(False)

        # Project gradient to the actionable features
        grad = grad * self.actionable_mask

        # Let the perturbation have epsilon magnitude
        delta = -grad / (torch.linalg.norm(grad, dim=-1, keepdims=True) + 1e-16) * self.epsilon

        return (x + delta).detach()

    def pgd_step(self, model, x, y, n_steps):
        """
        Obtain x_adv using Projected Gradient Ascent over the loss

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    n_steps: int, number of optimization steps

        Returns:    torch.Tensor, shape (B, D)
        """
        x_adv = torch.autograd.Variable(torch.clone(x), requires_grad=True)
        optimizer = torch.optim.Adam([x_adv], 0.1)

        for step in range(n_steps):
            optimizer.zero_grad()

            loss_x = -self.loss_function(model(x_adv), y)
            loss_x.backward()
            optimizer.step()

            # Project to L2 ball
            with torch.no_grad():
                delta = (x_adv - x) * self.unactionable_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon_AT
                delta[too_large] = delta[too_large] / (norm[too_large, None] + 1e-8) * self.epsilon_AT
                x_adv[:] = x + delta

        return x_adv.detach()