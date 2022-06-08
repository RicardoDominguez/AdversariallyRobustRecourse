"""
Plots Figure 3 of the paper, showing the magnitude of the minimum perturbation which invalidates recourse, for
standard Wachter-type minimum cost recourse.
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets and model types to include in the plot
n_seeds = 5
datasets = ['compas', 'adult', 'loan', 'german',  'bail']
model_types = ['lin', 'mlp']
title = 'Magnitude of min. perturbation invalidating recourse'
dataset_plot_names = {'compas': 'COMPAS', 'adult': 'Adult', 'loan': 'Loan', 'german': 'German',  'bail': 'Bail'}


# -----------------------------------------------------------------------------------------------------
# Load data to be plotted
# -----------------------------------------------------------------------------------------------------
norm_adv_pertbs = {}
median_norm_adv_pertbs = {}
for dataset in datasets:
    for model_type in model_types:
        # Stack the values of each seed
        adv_pertbs = np.zeros(0)
        for seed in range(n_seeds):
            fname = utils.get_metrics_save_dir(dataset, 'ERM', 0, model_type, 0, seed)
            vals = np.load(fname + '_advcost.npy')
            adv_pertbs = np.r_[adv_pertbs, np.log10(vals + 1e-9)]
        norm_adv_pertbs[dataset+model_type] = adv_pertbs
        median_norm_adv_pertbs[dataset+model_type] = np.percentile(adv_pertbs, [50])


# -----------------------------------------------------------------------------------------------------
# Plot the data
# -----------------------------------------------------------------------------------------------------

# Make the plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 11.0
})

plt.figure(figsize=(4, 1.6))
ax = plt.gca()

# Plot the violins
violin_positions = [1.1, 1.5, 3.4, 3.8, 5.4, 5.8, 7.3, 7.9, 9.4, 9.8]
parts = ax.violinplot([norm_adv_pertbs[dataset+model_type] for dataset in datasets for model_type in model_types],
                      showextrema=False,
                      widths=1,
                      positions=violin_positions)

colors = [sns.color_palette()[i] + (1,) for i in range(len(model_types))]
for i, pc in enumerate(parts['bodies']):
    pc.set_edgecolor('black')
    pc.set_facecolor(colors[i % len(model_types)])  # face color depends on the model type
    pc.set_alpha(0.7)

# Plot the medians
ax.scatter(violin_positions,
           [median_norm_adv_pertbs[dataset+model_type] for dataset in datasets for model_type in model_types],
           marker='o', color='k', s=30, zorder=3)

ax.scatter(violin_positions[::2],
           [median_norm_adv_pertbs[dataset+'lin'] for dataset in datasets],
           marker='.', color=colors[0], s=20, zorder=4)
# ax.scatter(violin_positions[1::2],
#            [median_norm_adv_pertbs[dataset+'mlp'] for dataset in datasets],
#            marker='.', color=colors[1], s=5, zorder=4)

# Formatting of the axis
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction='out')

ax.tick_params(axis='x', length=0)
ax.set_xticks([1.3, 3.6, 5.6, 7.6, 9.6])
ax.set_xticklabels([dataset_plot_names[dataset] for dataset in datasets])

ax.tick_params(axis='y', labelsize=11)
ax.set_yticks([-2, -4, -6, -8])
ax.set_yticklabels(['$10^{-2}$', '$10^{-4}$', '$10^{-6}$', '$10^{-8}$'])

ax.yaxis.grid(True)
ax.set_ylim([-6.5, 0])
ax.set_xlim([0, 10.8])

# Save and plot
plt.title(title, fontsize=11, x=0.45)
plt.tight_layout()
plt.savefig('figures/figure1.pdf', bbox_inches='tight')
# plt.show()
