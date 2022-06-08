"""
Generates and evaluates recourse for the trained decision-making classifiers.

Accepts the following parameters:
    - dataset (str)  ['adult', 'compas', 'german', 'loan', 'bail']
    - model (str)  either 'lin' for Linear Regression or 'mlp' for a neural network classifier
    - trainer (str)  one of ['ERM', 'AF', 'ALLR', 'ROSS']
    - lambda (float)  regularization strength (for ALLR and ROSS)
    - seed (int)  random seed
    - epsilon (float)  magnitude of uncertainty to robustify against
    - nexplain (int)  number of negatively classified individuals for which to compute recourse
"""

import utils
import data_utils
import trainers
import recourse
import attacks

import numpy as np
import torch
from tqdm import tqdm

def find_recourse_lin(model, trainer, scmm, X_explain, constraints, epsilon):
    # Equivalent robust classifer with modified decision threshold
    w, b = model.get_weights()
    Jw = w if scmm is None else scmm.get_Jacobian().T @ w
    dual_norm = np.sqrt(Jw.T @ Jw)
    b = b + dual_norm * epsilon

    explain = recourse.LinearRecourse(w, b)
    interv, recourse_valid, cost_recourse, _, interv_set = recourse.causal_recourse(X_explain, explain,
                                                                                       constraints, scm=scmm)
    return interv, interv_set, recourse_valid.astype(np.bool), cost_recourse


def find_recourse_mlp(model, trainer, scmm, X_explain, constraints, epsilon):
    hyperparams = utils.get_recourse_hyperparams(trainer)
    explain = recourse.DifferentiableRecourse(model, hyperparams)
    interv, recourse_valid, cost_recourse, _, interv_set = recourse.causal_recourse(X_explain, explain,
                                                                                    constraints, scm=scmm,
                                                                                    epsilon=epsilon, robust=epsilon>0)
    return interv, interv_set, recourse_valid.astype(np.bool), cost_recourse


def eval_recourse(dataset, model_type, trainer, random_seed, N_explain, epsilon, lambd, save_dir, save_adv=False):
    # Set the random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load the relevant dataset
    X, Y, constraints = data_utils.process_data(dataset)
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    # Load the relevant model
    model_dir = utils.get_model_save_dir(dataset, trainer, model_type, random_seed, lambd) + '.pth'
    model = trainers.LogisticRegression if model_type == 'lin' else trainers.MLP
    model = model(X_train.shape[-1], actionable_features=constraints['actionable'], actionable_mask=trainer=='AF')
    model.load_state_dict(torch.load(model_dir))
    model.set_max_mcc_threshold(X_train, Y_train)

    # Load the SCM
    scmm = utils.get_scm(model_type, dataset)

    # Sample N_explain number of misclassified points
    id_neg = model.predict(X_test) == 0
    X_neg = X_test[id_neg]
    N_Explain = min(N_explain, len(X_neg))
    id_explain = np.random.choice(np.arange(X_neg.shape[0]), size=N_Explain, replace=False)
    id_neg_explain = np.argwhere(id_neg)[id_explain]
    X_explain = X_neg[id_explain]

    # Find recourse
    find_recourse = find_recourse_lin if model_type == 'lin' else find_recourse_mlp
    interv, valid_interv_sets, recourse_valid, cost_recourse = find_recourse(model, trainer, scmm, X_explain,
                                                                             constraints, epsilon)
    print("Valid recourse: %.3f" % (recourse_valid.sum()/recourse_valid.shape[0]))
    print("Cost recourse: %.3f" % (cost_recourse[recourse_valid].mean()))

    np.save(save_dir + '_ids.npy', id_neg_explain)
    np.save(save_dir + '_valid.npy', recourse_valid)
    np.save(save_dir + '_cost.npy', cost_recourse)

    # Evaluate how fragile is the recourse recommendation
    if save_adv:
        attacker = attacks.CW_Adversary(scmm=scmm)
        _, valid_adv, cost_adv = attacker(model, X_explain[recourse_valid], torch.Tensor(interv[recourse_valid]),
                                          interv_set=valid_interv_sets)

        print("Valid adv: %.3f" % (valid_adv.sum() / recourse_valid.sum()))
        print("Less epsilon: %.3f" % ((cost_adv[valid_adv] < epsilon).sum() / recourse_valid.sum()))
        print("Cost adv: %.3f" % (cost_adv[valid_adv].mean()))
        np.save(save_dir + '_advcost.npy', cost_adv)
        np.save(save_dir + '_advvalid.npy', valid_adv)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['compas', 'bail', 'adult', 'german', 'loan'])
    parser.add_argument('--model', type=str, default='lin', choices=['lin', 'mlp'])
    parser.add_argument('--trainer', type=str, choices=['ERM', 'ALLR', 'AF', 'ROSS'])
    parser.add_argument('--lambd', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--nexplain', type=int, default=1000)

    args = parser.parse_args()
    save_name = utils.get_metrics_save_dir(args.dataset, args.trainer, args.lambd, args.model, args.epsilon, args.seed)
    eval_recourse(args.dataset, args.model, args.trainer, args.seed, args.nexplain, args.epsilon, args.lambd, save_name, True)
