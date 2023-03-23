# Optimal experimental design for staggered rollouts

This repository contains the code to reproduce the results presented in "optimal experimental design for staggered rollouts".

### Eamples to use the code are available at [`code`](code)

#### Nonadaptive experiments 

- [`Example`](code/Figure-2-4.ipynb) to solve T-optimal design and D-optimal design and generate Figures 2 and EC.1

- [`Example`](code/nonadaptive-flu.ipynb) to run synthetic nonadaptive experiments on empirical data and generate Figure 6

- [`Example`](code/compare-various-estimation-methods-designs.ipynb) to run synthetic nonadaptive experiments on empirical data with various benchmark designs, generate Figure 3, and generate the data for Figures EC.4 and EC.5

- [`Example`](code/Figure-10-11.ipynb) to generate Figures EC.4 and EC.5

#### Adaptive experiments

- [`Example`](code/adaptive_asymptotics-lemma-4.1.ipynb) to verify the finite sample properties of the asymptotic distributions derived in Lemma 4.1 and generate Figure EC.13

- [`Example`](code/adaptive_asymptotics-theorem-4.1.ipynb) to verify the finite sample properties of the asymptotic distributions derived in Theorem 4.1 and generate Figure EC.14 and EC.15, and Table EC.3

- [`Example`](code/adaptive-flu.ipynb) to run synthetic adaptive experiments using the Precision-Guided Adaptive Experiments (PGAE) algorithm on empirical data and generate Figures 7 and 8


### Empirical data sets used in this paper are available at [`data`](data)

- Data have been preprocessed into a matrix form

### Figures are available at [`result`](result)

### Code is available at [`code`](code) 

- ```utils_estimate.py```: within estimator, OLS, and GLS

- ```utils_carryover.py```: solve the optimal designs with carryover effects

- ```utils_design.py```: generate treatment designs and experimental data

- ```test_static.py```: run fixed-sample-size experiments and compare different treatment designs

- ```utils_static_covariate.py```: helper functions for nonadaptive experiments

- ```utils_adaptive.py```: Precision-Guided Adaptive Experiments (PGAE) algorithm

- ```utils_empirical.py```: run synthetic experiments on empirical data

- ```utils_import_data.py```: import empirical data

- ```utils_make_figures.py```: make figures

### Reference

```
@article{xiong2019optimal,
  title={Optimal experimental design for staggered rollouts},
  author={Xiong, Ruoxuan and Athey, Susan and Bayati, Mohsen and Imbens, Guido},
  journal={arXiv preprint arXiv:1911.03764},
  year={2019}
}
```
