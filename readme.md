# Adversarially Robust Recourse

This repository provides the code necessary to reproduce the experiments of the paper
["On the Adversarial Robustness of Causal Algorithmic Recourse"](/home/ricardo/Desktop/myrec2/data_utils.py).

## Prequisites

Install the packages in `requirements.txt`, for instance using

```
python -m venv myenv/
source myenv/bin/activate
pip install -r requirements.txt
```

## Reference implementations

### Our proposed method to generate adversarially robust recourse

Please refer to `evaluate_recourse.py` for an overview. The implementations are found in `recourse.py`. 

In the linear case, the method used to generate standard recourse is `LinearRecourse`. To generate robust recourse
against some uncertainty epsilon, we implement Equation 5 in the paper as 

```
w, b = model.get_weights()
Jw = w if scmm is None else scmm.get_Jacobian().T @ w
dual_norm = np.sqrt(Jw.T @ Jw)
b = b + dual_norm * epsilon
```

In the nonlinear case, please refer to `DifferentiableRecourse`.

### Our proposed ALLR regularizer

Please refer to `train_classifiers.py` for an overview. The implementations are found in `train_classifiers.py`.

In the linear setting, refer to `LogisticRegression`. In the non-linear setting, refer to `LLR_Trainer`.


## Plotting the figures of the paper

Since running the full set of experiments can be relatively time-consuming, we already provide the numerical results
of the experiments in the folder `results/`. To plot the results, simply run 

```
python plot_figure1.py     # plots Figure 3 in the paper
python plot_figure2.py     # plots Figure 4 in the paper
python plot_figures3_4.py  # plots Figure 5 and Figure 6 in the paper
```

Figures will be created in the folder `figures/`


## Running the experiments

Simply run 

```
python run_benchmarks.py --seed 0
python run_benchmarks.py --seed 1
python run_benchmarks.py --seed 2
python run_benchmarks.py --seed 3
python run_benchmarks.py --seed 4
```

If you wish to retrain the decision-making classifiers and the structural equations, simply delete the folders `models/`
and `scms/` before running `run_benchmarks.py`. Otherwise, the pretrained classifiers and SCMs will be used.

