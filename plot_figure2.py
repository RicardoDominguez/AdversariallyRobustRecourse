"""
Plots Figure 4 of the paper, showing the magnitude of the minimum perturbation which invalidates recourse for
recourse recommendations that have been robustified against epsilon uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

import utils
datasets = ['compas', 'german', 'adult', 'loan', 'bail']
model_types = ['lin', 'mlp']
epsilons = [0.001, 0.01, 0.1, 0.5]
save_dir = 'figures/'

# -----------------------------------------------------------------------------------------------------
# Load data to be plotted
# -----------------------------------------------------------------------------------------------------

norm_adv_pertbs = {model_type: {} for model_type in model_types}
for model_type in model_types:
    for dataset in datasets:
        for epsilon in epsilons:
            # Stack seeds
            all_pertbs = np.zeros(0)
            for seed in range(5):
                fname = utils.get_metrics_save_dir(dataset, 'ERM', 0, model_type, epsilon, seed)
                vals = np.load(fname + '_advcost.npy')
                all_pertbs = np.r_[all_pertbs, np.log10(vals)]

            if len(all_pertbs) > 0:
                norm_adv_pertbs[model_type][dataset+str(epsilon)] = all_pertbs
            else: # TODO: fix this
                norm_adv_pertbs[model_type][dataset + str(epsilon)] = np.zeros(1) + 0.5

# -----------------------------------------------------------------------------------------------------
# Plot the data
# -----------------------------------------------------------------------------------------------------

colors = [sns.color_palette()[i] + (1,) for i in range(len(datasets))]

def plot_violin(ax, data):
    data = [data[dataset+str(epsilon)] for epsilon in epsilons for dataset in datasets]
    parts = ax.violinplot(data, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_edgecolor('black')
        pc.set_facecolor(colors[i % len(datasets)])
        pc.set_alpha(0.7)

    for i in range(len(data)):
        ax.scatter(i+1, np.percentile(data[i], [50]),
                   marker='o', color=colors[i % len(datasets)], s=20, zorder=3, edgecolor='k')


def format(ax, title, title_loc):
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction='out')

    ax.tick_params(axis='x', length=0)
    ax.set_xticks([2.7, 7.7, 12.7, 17.7])
    ax.set_xticklabels(['$\epsilon=10^{-3}$', '$\epsilon=10^{-2}$', '$\epsilon=10^{-1}$', '$\epsilon=0.5$'])

    ax.tick_params(axis='y', labelsize=11)
    ax.set_yticks([-0.3, -1, -2, -3])
    ax.set_yticklabels([0.5, '$10^{-1}$', '$10^{-2}$', '$10^{-3}$'], fontsize=10)

    ax.yaxis.grid(True)
    ax.set_ylim([-3.2, 0])

    ax.set_xlabel('Magnitude of uncertainty $\epsilon$')
    ax.set_title(title, fontsize=11, loc=title_loc)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 11.0
})

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 2), sharey=True)

plot_violin(ax[0], norm_adv_pertbs['lin'])
plot_violin(ax[1], norm_adv_pertbs['mlp'])

format(ax[0], 'LR classifier', title_loc='left')
format(ax[1], 'NN classifier', title_loc='right')

fig.suptitle('Magnitude of min. perturbation invalidating robustified recourse', fontsize=11)
plt.tight_layout()

plt.savefig(save_dir+'figure2.pdf', bbox_inches='tight')
# plt.show()
