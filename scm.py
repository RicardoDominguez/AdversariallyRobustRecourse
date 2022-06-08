"""
This file contains the implementation of the Structural Causal Models used for modelling the effect of interventions
on the features of the individual seeking recourse.
"""

import os
import data_utils
import numpy as np
import torch

from itertools import chain, combinations  # for the powerset of actionable combinations of interventions


class SCM:
    """
    Includes all the relevant methods required for generating counterfactuals. Classes inheriting this class must
    contain the following objects:
        self.f: list of functions, each representing a structural equation. Function self.f[i] must have i+1 arguments,
                corresponding to X_1, ..., X_{i-1}, U_{i+1} each being a torch.Tensor with shape (N, 1), and returns
                the endogenous variable X_{i+1} as a torch.Tensor with shape (N, 1)
        self.inv_f: list of functions, corresponding to the inverse mapping X -> U. Each function self.inv_f[i] takes
                    as argument the features X as a torch.Tensor with shape (N, D), and returns the corresponding
                    exogenous variable U_{i+1} as a torch.Tensor with shape (N, 1)
        self.actionable: list of int, indices of the actionable features
        self.soft_interv: list of bool with len = D, indicating whether the intervention on feature soft_interv[i] is
                          modeled as a soft intervention (True) or hard intervention (False)
        self.mean: expectation of the features, such that when generating data we can standarize it
        self.std: standard deviation of the features, such that when generating data we can standarize it
    """
    def sample_U(self, N):
        """
        Return N samples from the distribution over exogenous variables P_U.

        Inputs:     N: int, number of samples to draw

        Outputs:    U: np.array with shape (N, D)
        """
        raise NotImplementedError

    def label(self, X):
        """
        Label the input instances X

        Inputs:     X: np.array with shape (N, D)

        Outputs:    Y:  np.array with shape (N, )
        """
        raise NotImplementedError

    def generate(self, N):
        """
        Sample from the observational distribution implied by the SCM

        Inputs:     N: int, number of instances to sample

        Outputs:    X: np.array with shape (N, D), standarized (since we train the models on standarized data)
                    Y: np.array with shape (N, )
        """
        U = self.sample_U(N).astype(np.float32)
        X = self.U2X(torch.Tensor(U))
        Y = self.label(X.numpy())
        X = (X - self.mean) / self.std

        return X.numpy(), Y

    def U2X(self, U):
        """
        Map from the exogenous variables U to the endogenous variables X by using the structural equations self.f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        X = []
        for i in range(U.shape[1]):
            X.append(self.f[i](*X[:i] + [U[:, [i]]]))
        return torch.cat(X, 1)

    def X2U(self, X):
        """
        Map from the endogenous variables to the exogenous variables by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        if self.inv_f is None:
            return X + 0.

        U = torch.zeros_like(X)
        for i in range(X.shape[1]):
            U[:, [i]] = self.inv_f[i](X)
        return U

    def counterfactual(self, Xn, delta, actionable=None, soft_interv=None):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    actionable: None or list of int, indices of the intervened upon variables
                    soft_interv: None or list of int, variables for which the interventions are soft (rather than hard)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        actionable = self.actionable if actionable is None else actionable
        soft_interv = self.soft_interv if soft_interv is None else soft_interv

        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            if i in actionable:
                if soft_interv[i]:
                    X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]])
                else:
                    X_cf.append(X[:, [i]] + delta[:, [i]])
            else:
                X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def counterfactual_batch(self, Xn, delta, interv_mask):
        """
        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    interv_sets: torch.Tensor (N, D)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        N, D = Xn.shape
        soft_mask = torch.Tensor(self.soft_interv).repeat(N, 1)
        hard_mask = 1. - soft_mask

        mask_hard_actionable = hard_mask * interv_mask
        mask_soft_actionable = soft_mask * interv_mask

        return self.counterfactual_mask(Xn, delta, mask_hard_actionable, mask_soft_actionable)


    def counterfactual_mask(self, Xn, delta, mask_hard_actionable, mask_soft_actionable):
        """
        Different way of computing counterfactuals, which may be more computationally efficient in some cases, specially
        if different instances have different actionability constrains, or hard/soft intervention criteria.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    mask_hard_actionable: torch.Tensor (N, D), 1 for actionable features under a hard intervention
                    mask_soft_actionable: torch.Tensor (N, D), 1 for actionable features under a soft intervention

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            X_cf.append((X[:, [i]] + delta[:, [i]]) * mask_hard_actionable[:, [i]] + (1 - mask_hard_actionable[:, [i]])
                        * (self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]] * mask_soft_actionable[:, [i]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def U2Xn(self, U):
        """
        Mapping from the exogenous variables U to the endogenous X variables, which are standarized

        Inputs:     U: torch.Tensor, shape (N, D)

        Outputs:    Xn: torch.Tensor, shape (N, D), is standarized
        """
        return self.X2Xn(self.U2X(U))

    def Xn2U(self, Xn):
        """
        Mapping from the endogenous variables X (standarized) to the exogenous variables U

        Inputs:     Xn: torch.Tensor, shape (N, D), endogenous variables (features) standarized

        Outputs:    U: torch.Tensor, shape (N, D)
        """
        return self.X2U(self.Xn2X(Xn))

    def Xn2X(self, Xn):
        """
        Transforms the endogenous features to their original form (no longer standarized)

        Inputs:     Xn: torch.Tensor, shape (N, D), features are standarized

        Outputs:    X: torch.Tensor, shape (N, D), features are not standarized
        """
        return Xn * self.std + self.mean

    def X2Xn(self, X):
        """
        Standarizes the endogenous variables X according to self.mean and self.std

        Inputs:     X: torch.Tensor, shape (N, D), features are not standarized

        Outputs:    Xn: torch.Tensor, shape (N, D), features are standarized
        """
        return (X - self.mean) / self.std

    def getActionable(self):
        """ Returns the indices of the actionable features, as a list of ints. """
        return self.actionable

    def getPowerset(self, actionable):
        """ Returns the power set of the set of actionable features, as a list of lists of ints. """
        s = actionable
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]

    def build_mask(self, mylist, shape):
        """
        Builds a torch.Tensor mask according to the list of indices contained in mylist. Used to build the masks of
        actionable features, or those of variables which are intervened upon with soft interventions.

        Inputs:     mylist: list(D) of ints or list(N) of lists(D) of ints, corresponding to indices
                    shape: list of ints [N, D]

        Outputs:    mask: torch.Tensor with shape (N, D), where mask[i, j] = 1. if j in mylist (for list of ints) or
                          j in mylist[i] (for list of list of ints)
        """
        mask = torch.zeros(shape)
        if type(mylist[0]) == list: # nested list
            for i in range(len(mylist)):
                mask[i, mylist[i]] = 1.
        else:
            mask[:, mylist] = 1.
        return mask

    def get_masks(self, actionable, shape):
        """
        Returns the mask of actionable features, actionable features which are soft intervened, and actionable
        features which are hard intervened.

        Inputs:     actionable: list(D) of int, or list(N) of list(D) of int, containing the indices of actionable feats
                    shape: list of int [N, D]

        Outputs:    mask_actionable: torch.Tensor (N, D)
                    mask_soft_actionable: torch.Tensor (N, D)
                    mask_hard_actionable: torch.Tensor (N, D)
        """
        mask_actionable = self.build_mask(actionable, shape)
        mask_soft = self.build_mask(list(np.where(self.soft_interv)[0]), shape)
        mask_hard_actionable = (1 - mask_soft) * mask_actionable
        mask_soft_actionable = mask_soft * mask_actionable
        return mask_actionable, mask_soft_actionable, mask_hard_actionable

class SCM_Loan(SCM):
    """ Semi-synthetic SCM inspired by the German Credit data set, introduced by Karimi et al. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f = [lambda U1: U1,
                  lambda X1, U2: -35 + U2,
                  lambda X1, X2, U3: -0.5 + 1 / (1 + torch.exp(1 - 0.5*X1 - 1 / (1 + torch.exp(-0.1*X2)) - U3)),
                  lambda X1, X2, X3, U4: 1 - 0.01*((X2-5)**2) + X1 + U4,
                  lambda X1, X2, X3, X4, U5: -1 + 0.1*X2 + 2*X1 + X4 + U5,
                  lambda X1, X2, X3, X4, X5, U6: -4 + 0.1*(X2+35) + 2*X1 + X1*X3 + U6,
                  lambda X1, X2, X3, X4, X5, X6, U7: -4 + 1.5*torch.clip(X6, 0, None) + U7
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]] + 35,
                      lambda X: torch.log(0.5 + X[:, [2]]) - torch.log(0.5 - X[:, [2]]) + 1 - 0.5*X[:, [0]]
                                - 1 / (1 + torch.exp(-0.1*X[:, [1]])),
                      lambda X: X[:, [3]] - 1 + 0.01*((X[:, [1]]-5)**2) - X[:, [0]],
                      lambda X: X[:, [4]] + 1 - 0.1*X[:, [1]] - 2*X[:, [0]] - X[:, [3]],
                      lambda X: X[:, [5]] + 4 - 0.1*(X[:, [1]]+35) - 2*X[:, [0]] - X[:, [0]]*X[:, [2]],
                      lambda X: X[:, [6]] + 4 - 1.5*torch.clip(X[:, [5]], 0, None)
                      ]

        self.mean = torch.Tensor([0, -4.6973433e-02, -5.9363052e-02,  1.3938685e-02,
                                    -9.7113004e-04,  4.8712617e-01, -2.0761824e+00])
        self.std = torch.Tensor([1, 11.074237, 0.13772593, 2.787965, 4.545642, 2.5124693, 5.564847])

        self.actionable = [2, 5, 6]
        self.soft_interv = [True, True, False, True, True, False, False]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.5, N)
        U2 = np.random.gamma(10, 3.5, N)
        U3 = np.random.normal(0, np.sqrt(0.25), N)
        U4 = np.random.normal(0, 2, N)
        U5 = np.random.normal(0, 3, N)
        U6 = np.random.normal(0, 2, N)
        U7 = np.random.normal(0, 5, N)
        return np.c_[U1, U2, U3, U4, U5, U6, U7]

    def label(self, X):
        L, D, I, S = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
        p = 1 / (1 + np.exp(-0.3*(-L - D + I + S + I*S)))
        Y = np.random.binomial(1, p)
        return Y

class IMF(SCM):
    """ SCM corresponding to the Independently Manipulable Feautes assumption (i.e. no causal relations) """
    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)

        self.f = []
        for i in range(N):
            self.f.append(lambda *args: args[-1]) # f(X_1, ..., X_i, U_{i+1}) = U_{i+1}

        self.inv_f = None

        self.mean = torch.zeros(N)
        self.std = torch.ones(N)

        self.actionable = list(np.arange(N))
        self.soft_interv = [True]*N

# ----------------------------------------------------------------------------------------------------------------------
# The following functions are to fit the structural equations using MLPs with 1 hidden layer, in the case where the
# causal graph is know but the structural equations are unknown.
# ----------------------------------------------------------------------------------------------------------------------

class MLP1(torch.nn.Module):
    """ MLP with 1-layer and tanh activation function, to fit each of the structural equations """
    def __init__(self, input_size, hidden_size=100):
        """
        Inputs:     input_size: int, number of features of the data
                    hidden_size: int, number of neurons for the hidden layer
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)
        self.activ = torch.nn.Tanh()

    def forward(self, x):
        """
        Inputs:     x: torch.Tensor, shape (N, input_size)

        Outputs:    torch.Tensor, shape (N, 1)
        """
        return self.linear2(self.activ(self.linear1(x)))


class SCM_Trainer:
    """ Class used to fit the structural equations of some SCM """
    def __init__(self, batch_size=100, lr=0.001, print_freq=100, verbose=False):
        """
        Inputs:     batch_size: int
                    lr: float, learning rate (Adam used as the optimizer)
                    print_freq: int, verbose every print_freq epochs
                    verbose: bool
        """
        self.batch_size = batch_size
        self.lr = lr
        self.print_freq = print_freq
        self.loss_function = torch.nn.MSELoss(reduction='mean') # Fit using the Mean Square Error
        self.verbose = verbose

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs):
        """
        Inputs:     model: torch.nn.Model
                    X_train: torch.Tensor, shape (N, D)
                    Y_train: torch.Tensor, shape (N, 1)
                    X_test: torch.Tensor, shape (M, D)
                    Y_test: torch.Tensor, shape (M, 1)
                    epochs: int, number of training epochs
        """
        X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test)
        train_dst = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        test_dst = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dst, batch_size=1000, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        accuracies = np.zeros(int(epochs / self.print_freq))

        prev_loss = np.inf
        val_loss = 0
        for epoch in range(epochs):
            if self.verbose:
                if epoch % self.print_freq == 0:
                    mse = self.loss_function(model(X_test), Y_test)
                    print("Epoch: {}. MSE {}.".format(epoch, mse))

            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self.loss_function(model(x), y)
                loss.backward()
                optimizer.step()


class Learned_Adult_SCM(SCM):
    """
    SCM for the Adult data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Inputs:
        - linear: whether to fit linear or non-linear structural equations
    """
    def __init__(self, linear=False):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(6)
        self.std = torch.ones(6)

        self.actionable = [4, 5]
        self.soft_interv = [True, True, True, True, False, False]

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(3, 1), torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)
        return MLP1(3), MLP1(4), MLP1(4)

    def get_Jacobian(self):
        assert self.linear, "Jacobian only used for linear SCM"

        w4 = self.f1.weight[0]
        w5 = self.f2.weight[0]
        w6 = self.f3.weight[0]

        w41, w42, w43 = w4[0].item(), w4[1].item(), w4[2].item()
        w51, w52, w53, w54 = w5[0].item(), w5[1].item(), w5[2].item(), w5[3].item()
        w61, w62, w63, w64 = w6[0].item(), w6[1].item(), w6[2].item(), w6[3].item()

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [w41, w42, w43, 1, 0, 0],
                         [w51 + w54*w41, w52 + w54*w42, w53 + w54*w43, w54, 1, 0],
                         [w61 + w64*w41, w62 + w64*w42, w63 + w64*w43, w64, 0, 1]])

    def get_Jacobian_interv(self, interv_set):
        """ Get the Jacobian of the structural equations under some interventions """
        J = self.get_Jacobian()
        for i in range(J.shape[0]):
            # If we are hard intervening, do not allow changes from upstream causal effects (set to 0)
            if i in interv_set and not self.soft_interv[i]:
                for j in range(i):
                    J[i][j] = 0.
        return J


    def fit_eqs(self, X, save=None):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        if os.path.isfile(save+model_type+'_f1.pth'):
            print('Fitted SCM already exists')
            return


        mask_1 = [0, 1, 2]
        mask_2 = [0, 1, 2, 3]
        mask_3 = [0, 1, 2, 3]

        f1, f2, f3 = self.get_eqs()

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        train_epochs = 10
        trainer = SCM_Trainer(verbose=False, print_freq=1, lr=0.005)
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 3].reshape(-1, 1), train_epochs)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 4].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 4].reshape(-1, 1), train_epochs)
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 5].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 5].reshape(-1, 1), train_epochs)

        if save is not None:

            torch.save(f1.state_dict(), save+model_type+'_f1.pth')
            torch.save(f2.state_dict(), save+model_type+'_f2.pth')
            torch.save(f3.state_dict(), save+model_type+'_f3.pth')

        self.set_eqs(f1, f2, f3) # Build the structural equations

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2, f3 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth'))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth'))
        f3.load_state_dict(torch.load(name + model_type + '_f3.pth'))

        self.set_eqs(f1, f2, f3)

    def set_eqs(self, f1, f2, f3):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1, self.f2, self.f3 = f1, f2, f3

        self.f = [lambda U1: U1,
                  lambda X1, U2: U2,
                  lambda X1, X2, U3: U3,
                  lambda X1, X2, X3, U4: f1(torch.cat([X1, X2, X3], 1)) + U4,
                  lambda X1, X2, X3, X4, U5: f2(torch.cat([X1, X2, X3, X4], 1)) + U5,
                  lambda X1, X2, X3, X4, X5, U6: f3(torch.cat([X1, X2, X3, X4], 1)) + U6,
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]],
                      lambda X: X[:, [2]],
                      lambda X: X[:, [3]] - f1(X[:, [0,1,2]]),
                      lambda X: X[:, [4]] - f2(X[:, [0,1,2,3]]),
                      lambda X: X[:, [5]] - f3(X[:, [0,1,2,3]]),
                      ]

class Learned_COMPAS_SCM(SCM):
    """
    SCM for the COMPAS data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Age, Gender -> Race, Priors
    Race -> Priors
    Feature names: ['age', 'isMale', 'isCaucasian', 'priors_count']
    """
    def __init__(self, linear=False):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(4)
        self.std = torch.ones(4)

        self.actionable = [3]
        self.soft_interv = [True, True, True, False]

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(2, 1), torch.nn.Linear(3, 1)
        return MLP1(2), MLP1(3)

    def get_Jacobian(self):
        assert self.linear, "Jacobian only used for linear SCM"

        w3 = self.f1.weight[0]
        w4 = self.f2.weight[0]

        w31, w32 = w3[0].item(), w3[1].item()
        w41, w42, w43 = w4[0].item(), w4[1].item(), w4[2].item()

        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [w31, w32, 1, 0],
                         [w41 + w43*w31, w42 + w43*w32, w43, 1]])

    def get_Jacobian_interv(self, interv_set):
        """ Get the Jacobian of the structural equations under some interventions """
        J = self.get_Jacobian()
        for i in range(J.shape[0]):
            # If we are hard intervening, do not allow changes from upstream causal effects (set to 0)
            if i in interv_set and not self.soft_interv[i]:
                for j in range(i):
                    J[i][j] = 0.
        return J

    def fit_eqs(self, X, save=None):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X3 = f1(X1, X2, U3)
            X4 = f2(X1, X2, X3, U4)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        if os.path.isfile(save+model_type+'_f1.pth'):
            print('Fitted SCM already exists')
            return

        mask_1 = [0, 1]
        mask_2 = [0, 1, 2]

        f1, f2 = self.get_eqs()

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        trainer = SCM_Trainer(verbose=False, print_freq=1, lr=0.005)
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 2].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 2].reshape(-1, 1), 50)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 3].reshape(-1, 1), 50)

        if save is not None:
            torch.save(f1.state_dict(), save+model_type+'_f1.pth')
            torch.save(f2.state_dict(), save+model_type+'_f2.pth')

        self.set_eqs(f1, f2)  # Build the structural equations

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth'))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth'))

        self.set_eqs(f1, f2)

    def set_eqs(self, f1, f2):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1 = f1
        self.f2 = f2

        self.f = [lambda U1: U1,
                  lambda X1, U2: U2,
                  lambda X1, X2, U3: f1(torch.cat([X1, X2], 1)) + U3,
                  lambda X1, X2, X3, U4: f2(torch.cat([X1, X2, X3], 1)) + U4,
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]],
                      lambda X: X[:, [2]] - f1(X[:, [0, 1]]),
                      lambda X: X[:, [3]] - f2(X[:, [0, 1, 2]]),
                      ]


def generate_SCM_data(id, N):
    """
    Return samples of the SCM (if synthetic), as well as information pertaining to the features (which ones are
    actionable, increasing, decreasing, and categorical)
    
    Inputs:     id: str, dataset id. One of 'German', 'Adult'.
                N: int, number of samples to draw (if the data set is synthetic).
                
    Outputs:    myscm: type SCM
                X: np.array (N, D) or None
                Y: np.array (N, ) or None
                actionable: list of ints, indices of the actionable features
                increasing: list of ints, indices of the features which can only be increased (actionability constrain)
                decreasing: list of ints, indices of the features which can only be decreased (actionability constrain)
                categorical: list of ints, indices of the features which are categorical (and thus not real-valued)
    """
    if id == 'German':
        myscm = SCM_Loan()
    elif id == 'Adult':
        myscm = Learned_Adult_SCM()
        myscm.load('scms/adult_scm')
    else:
        raise NotImplemented

    if id == 'German': # synthetic, generate the data
        X, Y = myscm.generate(N)
    else: # real world data set, no data returned
        X, Y = None, None

    actionable = myscm.getActionable()
    if id == 'German':
        increasing = [2]
        decreasing = []
        categorical = [0]
    elif id == 'Adult':
        increasing = [4, 5]
        decreasing = []
        categorical = [0, 1, 2]

    return myscm, X, Y, actionable, increasing, decreasing, categorical