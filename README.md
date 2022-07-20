# Optimal experimental design for staggered rollouts

### Code is available at code/

- utils_estimate.py: code for within estimator, OLS, and GLS

- utils_carryover.py: code to solve the optimal designs with carryover effects

- utils_design.py: code to generate treatment designs and experimental data

- test_static.py: code to run fixed-sample-size experiments and compare different treatment designs

- utils_adaptive.py: code for Precision-Guided Adaptive Experiments (PGAE) algorithm

- utils_empirical.py: helper functions to run synthetic experiments on empirical data

- utils_import_data.py: helper functions to import empirical data

- utils_make_figures.py: helper functions to make figures

### Eamples to use the code are available at code/

#### Fixed-sample-size experiments 

- Figure-2-4.ipynb: solve T-optimal design and D-optimal design

- compare-various-estimation-methods-designs.ipynb: run synthetic fixed-sample-size experiments on empirical data

- Figure-10-11.ipynb: make Figures 10 and 11

#### Sequential experiments

- adaptive_asymptotics-lemma-4.1.ipynb: verify the finite sample properties of the asymptotic distributions derived in Lemma 4.1

- adaptive_asymptotics-theorem-4.1.ipynb: verify the finite sample properties of the asymptotic distributions derived in Theorem 4.1

- adaptive-flu.ipynb: run synthetic sequential experiments on empirical data


### Empirical data sets used in this paper are available at data/

- Data have been preprocessed into a matrix form

### Figures are available at result/
