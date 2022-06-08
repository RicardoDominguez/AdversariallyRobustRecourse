import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

import utils

datasets = ['compas', 'adult', 'loan', 'german', 'bail']
model_types = ['lin', 'mlp']
trainers = {'lin': ['ERM', 'ALLR', 'ROSS', 'AF'],
            'mlp': ['ERM', 'ALLR', 'ROSS', 'AF', 'ALLR0', 'ALLR1']}

epsilons = [0, 0.1]
save_dir = 'figures/'

# -----------------------------------------------------------------------------------------------------
# Load data to be plotted
# -----------------------------------------------------------------------------------------------------
n_seeds = 5
max_value = 1.0
data = {}
for dataset in datasets:
    for model_type in model_types:
        max_cost = -np.inf
        for trainer in trainers[model_type]:
            data_dict = {'eps': [], 'valid': [], 'cost': []}

            # Hyperparameter used in each case
            lambd = utils.get_lambdas(dataset, model_type, trainer)

            for epsilon in epsilons:
                cost, valid, mcc, acc = 0, 0, 0, 0
                for seed in range(n_seeds):
                    if epsilon == 0:
                        fname = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, 0, seed)
                        mcc += float(np.load(fname + '_mccs.npy'))
                        acc += float(np.load(fname + '_accs.npy'))

                    # Compare with valid recourse found using ERM
                    fname = utils.get_metrics_save_dir(dataset, 'ERM', lambd, model_type, epsilon, seed)
                    erm_ids = np.load(fname + '_ids.npy')
                    erm_v = np.load(fname + '_valid.npy').astype(np.bool)

                    fname = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, epsilon, seed)
                    ids = np.load(fname + '_ids.npy')
                    v = np.load(fname + '_valid.npy').astype(np.bool)
                    c = np.load(fname + '_cost.npy')

                    # Find the individuals for which recourse was found for both ERM and the considered method.
                    # This ensures that we only compare cost of recourse between the same individuals.
                    # That is, compared to the recourse offered by ERM, how does the recourse cost change for those
                    # individuals by adopting the proposed method?
                    ids_both = np.array([i in erm_ids for i in ids])

                    if model_type == 'mlp':
                        # Discard as valid if we were able to find an adversarial perturbation. We only do this for mlps
                        # as for linear models we are able to solve the problem with large precision.
                        rv = np.load(fname + '_advcost.npy')
                        if epsilon > 0:
                            valid_rv = rv > epsilon - 1e-3  # allow 1% leeway for numerical error, note that eps=0.1
                        else:
                            valid_rv = rv > -np.inf  # all of them are valid cfs

                        if v.sum() > 0:
                            valid += np.sum(valid_rv) / v.shape[0]
                            # cost += c[v.astype(np.bool)][valid_rv.astype(np.bool)].mean()

                            v[np.argwhere(v)[np.logical_not(valid_rv)]] = False
                            v = np.logical_and(ids_both, v)
                            cost += c[v].mean()

                    else:
                        if v.sum() > 0:
                            valid += np.sum(v) / v.shape[0]
                            v = np.logical_and(ids_both, v)
                            cost += c[v].mean()

                if valid > 0:
                    data_dict['eps'].append(epsilon)
                    data_dict['cost'].append(cost/n_seeds*max_value)
                    max_cost = max(max_cost, cost/n_seeds)
                    data_dict['valid'].append(valid/n_seeds*max_value)

                    if epsilon == 0:
                        data_dict['ccw'] = mcc / n_seeds * max_value
                        data_dict['acc'] = acc / n_seeds * max_value

            data[dataset + model_type + trainer] = data_dict

        # Normalize cost across trainers such that the maximum is 1
        for trainer in trainers[model_type]:
            data[dataset + model_type + trainer]['cost'] /= max_cost

# -----------------------------------------------------------------------------------------------------
# Plot the data
# -----------------------------------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 11.0
})

dataset_plot_names = {'compas': 'COMPAS', 'adult': 'Adult', 'loan': 'Loan', 'german': 'German',  'bail': 'Bail'}
train_plot_names = {'ERM': 'ERM', 'ALLR': 'ALLR', 'ROSS': 'Ross', 'AF': 'AF', 'ALLR1': '$\mu_1=0$', 'ALLR0': '$\mu_2=0$'}

colors = [sns.color_palette()[i] + (1,) for i in range(9)]

def make_plot(model_type, trainers, title, save_name, lsize=8):
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(8, 4), sharey=True)
    fig.suptitle(title, fontsize=11, y=1.)

    for id, dataset in enumerate(datasets):
        # Set title of the dataset being plotted
        axs[0, id].set_title(dataset_plot_names[datasets[id]], fontsize=11)

        # First two rows of plots, plotting % valid recourse and cost of recourse
        offset = [-0.015, -0.005, 0.005, 0.015]
        for it, trainer in enumerate(trainers):
            data_dict = data[dataset + model_type + trainer]
            offset_eps = [eps + offset[it] for eps in data_dict['eps']]
            axs[0, id].plot(offset_eps, data_dict['valid'], '--', label='Valid'+trainer, c=colors[it], alpha=0.4, zorder=-1)
            axs[1, id].plot(offset_eps, data_dict['cost'],  '--', label='Cost'+trainer, c=colors[it], alpha=0.4, zorder=-1)
            axs[0, id].plot(offset_eps, data_dict['valid'], 'o', label='Valid' + trainer, c=colors[it], markersize=5, zorder=10)
            axs[1, id].plot(offset_eps, data_dict['cost'], 'o', label='Cost' + trainer, c=colors[it], markersize=5, zorder=10)

        # Last row of plots, plotting CCW score and accuracy
        accs = [data[dataset + model_type + trainer]['acc'] for trainer in trainers]
        ccws = [data[dataset + model_type + trainer]['ccw'] for trainer in trainers]

        b = axs[2, id].bar(np.arange(len(accs)), accs, width=0.3, align='center', color=colors[4])
        axs[2, id].bar_label(b, fontsize=7,  labels=[str(int((np.rint(a*100)))) for a in accs])
        b = axs[2, id].bar(np.arange(len(ccws)) + 0.3, ccws, width=0.3, align='center', color=colors[8])
        axs[2, id].bar_label(b, fontsize=7, labels=[str(int((np.rint(a*100)))) for a in ccws])

        # For the first two rows of plots, the x-axis is the epsilon
        for i in range(2):
            axs[i, id].set_xticks([0.005, 0.095])
            axs[i, id].set_xlim([-0.03, 0.13])
            axs[i, id].set_xticklabels(['$\epsilon=0$', '$\epsilon=0.1$'], fontsize=10)
            axs[i, id].set_yticks(np.array([0, 0.25, 0.50, 0.75, 1.00])*max_value)

        # Y-axis in the grid
        for i in range(3):
            axs[i, id].yaxis.grid(True)

        # Names of the trainer along x-axis of last plot
        axs[2, id].set_xticks([0.15, 1.15, 2.2, 3.15])
        axs[2, id].set_xticklabels([train_plot_names[trainer] for trainer in trainers], fontsize=lsize)

        for i in range(3):
            axs[i, id].tick_params(axis=u'x', which=u'both', length=0)


    # Y-axis size
    plt.ylim(np.array([-0.1, 1.1])*max_value)
    axs[0, 0].set_ylabel('\% recourse \n found', fontsize=11)
    axs[1, 0].set_ylabel('Relative cost \n of recourse', fontsize=11)
    axs[2, 0].set_ylabel('Prediction \n performance', fontsize=11)

    # Save
    plt.tight_layout()
    plt.savefig(save_dir+save_name+'.pdf')
    # plt.show()

make_plot('lin', ['ERM', 'ALLR', 'ROSS', 'AF'], 'LR classifiers', 'figure3')
make_plot('mlp', ['ERM', 'ALLR', 'ROSS', 'AF'], 'NN classifiers', 'figure4')
make_plot('mlp', ['ERM', 'ALLR', 'ALLR1', 'ALLR0'], 'Ablation study - NN classifiers', 'figure5', lsize=6)
