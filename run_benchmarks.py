"""
Reproduces the experiments of the paper "On the Adversarial Robustness of Causal Algorithmic Recourse"
"""

import os
import numpy as np
import torch

import scm
import utils
import data_utils
import train_classifiers
import evaluate_recourse


def run_benchmark(models, datasets, seed, N_explain):
    # Here we will store the models and the experimental results, respectively
    dirs_2_create = [utils.model_save_dir, utils.metrics_save_dir, utils.scms_save_dir]
    for dir in dirs_2_create:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # ------------------------------------------------------------------------------------------------------------------
    # FIT THE STRUCTURAL CAUSAL MODELS
    # ------------------------------------------------------------------------------------------------------------------
    learned_scms = {'adult': scm.Learned_Adult_SCM, 'compas': scm.Learned_COMPAS_SCM}
    for dataset in datasets:
        for model_type in ['lin', 'mlp']:
            # We only need to learn the SCM for some of the data sets
            if dataset in learned_scms.keys():
                # Check if the structural equations have already been fitted
                print('Fitting SCM for %s...' % (dataset))

                # Learn a single SCM (no need for multiple seeds)
                np.random.seed(0)
                torch.manual_seed(0)

                X, _, _ = data_utils.process_data(dataset)
                myscm = learned_scms[dataset](linear=model_type=='lin')
                myscm.fit_eqs(X.to_numpy(), save=utils.scms_save_dir + dataset)

    # ------------------------------------------------------------------------------------------------------------------
    # TRAIN THE DECISION-MAKING CLASSIFIERS
    # ------------------------------------------------------------------------------------------------------------------
    trainers = ['ERM', 'AF', 'ALLR', 'ROSS', 'ALLR0', 'ALLR1']
    for model_type in models:
        for trainer in trainers:
            for dataset in datasets:
                if not (model_type == 'lin' and len(trainer) > 4):  # abalation only for mlp
                    lambd = utils.get_lambdas(dataset, model_type, trainer)
                    save_dir = utils.get_model_save_dir(dataset, trainer, model_type, seed, lambd)

                    # Train the model if it has not been already trained
                    if not os.path.isfile(save_dir+'.pth'):
                        print('Training... %s %s %s' % (model_type, trainer, dataset))
                        train_epochs = utils.get_train_epochs(dataset, model_type, trainer)
                        accuracy, mcc = train_classifiers.train(dataset, trainer, model_type, train_epochs, lambd, seed,
                                                                verbose=True, save_dir=save_dir)

                        # Save the performance metrics of the classifier
                        save_name = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, 0, seed)
                        np.save(save_name + '_accs.npy', np.array([accuracy]))
                        np.save(save_name + '_mccs.npy', np.array([mcc]))
                        print(save_name + '_mccs.npy')
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # EVALUATE RECOURSE FOR THE TRAINED CLASSIFIERS
    # # ------------------------------------------------------------------------------------------------------------------
    def run_evaluation(dataset, model_type, trainer, seed, N_explain, epsilon, save_adv):
        print('Evaluating... %s %s %s eps: %.3f' % (model_type, trainer, dataset, epsilon))
        lambd = utils.get_lambdas(dataset, model_type, trainer)
        save_name = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, epsilon, seed)
        evaluate_recourse.eval_recourse(dataset, model_type, trainer, seed, N_explain, epsilon, lambd, save_name, save_adv)
    # #
    # For the first experiment, we consider the fragility of recourse for epsilon = 0
    trainer = 'ERM'
    epsilon = 0
    for model_type in models:
        for dataset in datasets:
            run_evaluation(dataset, model_type, trainer, seed, N_explain, epsilon, True)

    # For the second experiment, we assess the effectiveness in robustifying against uncertainty epsilon > 0
    trainer = 'ERM'
    epsilons = [0.001, 0.01, 0.1, 0.5]
    for model_type in models:
        for dataset in datasets:
            for epsilon in epsilons:
                run_evaluation(dataset, model_type, trainer, seed, N_explain, epsilon, True)

    # # For the third experiment, we compare against classifiers trained using AF, ROSS and ALLR
    epsilons = [0., 0.1]
    trainers = ['AF', 'ROSS', 'ALLR', 'ALLR0', 'ALLR1']
    for model_type in models:
        for dataset in datasets:
            for trainer in trainers:
                if not (model_type == 'lin' and len(trainer) > 4):
                    for epsilon in epsilons:
                        run_evaluation(dataset, model_type, trainer, seed, N_explain, epsilon, model_type=='mlp')


if __name__ == "__main__":
    models = ['lin', 'mlp']
    datasets = ['german', 'compas', 'adult', 'loan', 'bail']
    N_explain = 1000  # number of points for which recourse is found

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_benchmark(models, datasets, args.seed, N_explain)

