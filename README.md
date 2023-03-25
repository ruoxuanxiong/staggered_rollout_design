# Optimal experimental design for staggered rollouts

This repository contains the code to reproduce the results presented in "optimal experimental design for staggered rollouts"

### Eamples to use the code are available at [`code`](code)

#### Nonadaptive experiments 

- [`Example`](code/optimal-design-Figure-2-4-EC1.ipynb) to solve T-optimal design and D-optimal design and generate Figures 2 and EC.1

- [`Example`](code/nonadaptive-flu-Figure-6.ipynb) to run synthetic nonadaptive experiments on empirical data and generate Figure 6

- [`Example`](code/compare-estimator-design-Figure-EC4-EC5.ipynb) to run synthetic nonadaptive experiments on empirical data with various benchmark designs and specifications, and generate the data for Figures EC.4 and EC.5


#### Adaptive experiments

- [`Example`](code/lemma-4.1-finite-sample-Figure-EC13.ipynb) to verify the finite sample properties of the asymptotic distributions derived in Lemma 4.1 and generate Figure EC.13

- [`Example`](code/theorem-4.1-finite-sample-Figure-EC14-15.ipynb) to verify the finite sample properties of the asymptotic distributions derived in Theorem 4.1 and generate Figure EC.14 and EC.15, and Table EC.3

- [`Example`](code/adaptive-flu-Figure-7-8.ipynb) to run synthetic adaptive experiments using the Precision-Guided Adaptive Experiments (PGAE) algorithm on empirical data and generate Figures 7 and 8

#### Generate illustrative figures 

- [`Example`](code/carryover-effect-Figure-1.ipynb) to generate two examples of dynamics of carryover effects in Figure 1

- [`Example`](code/illustrate-designs-Figure-3.ipynb) to generate various treatment designs in Figure 3

### Results on nonadaptive and adaptive experiments are available at [`result`](result)

### Illustrative figures are at [`figures`](figures)

### Empirical data sets used in this paper are available at [`data`](data)

- Data have been preprocessed into a matrix form



### Helper functions are available at [`code`](code) 

- ```utils_estimate.py```: OLS and GLS

- ```utils_carryover.py```: solve the optimal designs with carryover effects

- ```utils_design.py```: generate treatment designs and experimental data

- ```utils_nonadaptive.py```: helper functions for nonadaptive experiments

- ```utils_adaptive.py```: helper functions for adaptive experiments and Precision-Guided Adaptive Experiments (PGAE) algorithm

- ```utils_empirical.py```: run synthetic experiments on empirical data

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
