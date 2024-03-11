# Optimal experimental design for staggered rollouts

This repository contains the Python code to reproduce the results presented in [optimal experimental design for staggered rollouts](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3483934)

### Quickstart

To reproduce the results, download this repo on a machine with Python, run each of Jupyter Notebooks in the [`code`](code) without modification, and then the results are saved in [`result`](result). All the notebooks can be run standalone. To run the notebooks, you do not need to set any pathnames; everything is relative. Only standard libraries (Numpy, pandas, SciPy, Matplotlib, and seaborn) are required in the notebooks.

### Solve optimal design

- Run [`notebook`](code/optimal-design-Figure-2-4-EC1.ipynb) to solve

  - fraction of treated units per period in the T-optimal design ([`Figure 2`](figures/carryover-t-optimal.pdf))
  - fraction of treated units per period for the D-optimal design ([`Figure EC.1`](figures/carryover-d-optimal.pdf))
  - optimal fraction of treated units per period to maximize the precision of each of the estimated instantaneous and lagged effects ([`Figure 4`](figures/carryover-t-optimal-s-curve.pdf))


### Run nonadaptive experiments

- Run [`notebook`](code/nonadaptive-flu-Figure-6.ipynb)

  - run synthetic nonadaptive experiments on the flu data for 2,000 iterations 
  - compare various treatment designs, including benchmark designs, linear staggered design (optimal when $\ell = 0$), nonlinear staggered design (optimal for general $\ell$), stratified nonlinear staggered design) 
  - generate [`Figure 6`](result/flu/flu_T_7_varying_N_lag_2_agg.pdf)

- Run [`notebook`](code/compare-estimator-design-Figure-EC4-EC5.ipynb) 

  - run synthetic nonadaptive experiments on the flu data for 1,000 iterations 
  - compare various outcome specifications, including without fixed effects, with unit fixed effect only, with time fixed effect only, with two-way fixed effects, and with two-way fixed effects and latent covariates
  - compare various treatment designs
  - generate [`Figure EC.4`](result/flu/flu_N_25_T_7_various_methods-full.pdf) and [`Figure EC.5`](result/flu/flu_N_25_T_7_bias-variance.pdf) 

### Run adaptive experiments

- Run [`notebook`](code/adaptive-flu-Figure-7-8.ipynb)

  - run synthetic adaptive experiments using the Precision-Guided Adaptive Experiments (PGAE) algorithm on the flu data for 10,000 iterations
  - generate [`Figure 7`](result/flu-adaptive/flu_termination_time.pdf) and [`Figure 8`](result/flu-adaptive/flu_adaptive_comparison.pdf)

- Run [`notebook`](code/lemma-4.1-finite-sample-Figure-EC13.ipynb)

  - verify the finite sample properties of the asymptotic distributions derived in Lemma 4.1
  - generate [`subplots in Figure EC.13`](result/simulation/)

- Run [`notebook`](code/theorem-4.1-finite-sample-Figure-EC14-15.ipynb)

  - verify the finite sample properties of the asymptotic distributions derived in Theorem 4.1 
  - generate [`subplots in Figure EC.14 and EC.15, and Table EC.3`](result/simulation/)


### Generate illustrative figures 

- Run [`notebook`](code/carryover-effect-Figure-1.ipynb)

  - generate two examples of dynamics of carryover effects in Figure 1 ([`Example 1`](figures/cumulative_effect_new_infection.pdf) and [`Example 2`](figures/wearout_effect_app.pdf))

- Run [`notebook`](code/illustrate-designs-Figure-3.ipynb)
  
  - generate various treatment designs in Figure 3 (various designs are stored [`here`](figures/))


### Empirical data sets used in this paper are available at [`data`](data)

- Data have been preprocessed into a matrix form

### Helper functions are collected in [`code`](code) 

The following scripts collect all the helper functions used to solve treatment designs and run synthetic experiments. The helper functions are called in the notebooks listed above. You do not need to separately run any of the scripts to replicate the results in the paper. 

- ```utils_estimate.py```: within transformation, OLS and GLS

- ```utils_carryover.py```: solve the optimal designs with carryover effects

- ```utils_design.py```: generate treatment designs and experimental data

- ```utils_nonadaptive.py```: helper functions for nonadaptive experiments

- ```utils_adaptive.py```: helper functions for adaptive experiments and Precision-Guided Adaptive Experiments (PGAE) algorithm

- ```utils_empirical.py```: helper functions to pre-process empirical data that are used to run synthetic experiments

- ```utils_import_data.py```: import empirical data

### Reference

```
@article{xiong2023optimal,
  title={Optimal experimental design for staggered rollouts},
  author={Xiong, Ruoxuan and Athey, Susan and Bayati, Mohsen and Imbens, Guido},
  journal={Management Science, accepted},
  year={2023}
}
```
