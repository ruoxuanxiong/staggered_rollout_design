# Optimal experimental design for staggered rollouts

Code is available in code/

- utils_estimate.py: code for within estimator, OLS, and GLS

- utils_carryover.py: code to solve the optimal designs with carryover effects

- utils_design.py: code to generate treatment designs and experimental data

- test_static.py: code to run fixed-sample-size experiments and compare different treatment designs

- utils_adaptive.py: code for Precision-Guided Adaptive Experiments (PGAE) algorithm

- utils_empirical.py: helper functions to run synthetic experiments on empirical data

- utils_import_data.py: helper functions to import empirical data

- utils_make_figures.py: helper functions to make figures

Eamples to use the code are available in code/

- See Figure-2-4.ipynb for examples to solve T-optimal design and D-optimal design

- See Figure-3-7.ipynb for examples to run synthetic fixed-sample-size experiments on empirical data

- See adaptive_asymptotics-lemma-4.1.ipynb and adaptive_asymptotics-theorem-4.1.ipynb for examples to verify the finite sample properties of the asymptotic distributions derived in Lemma 4.1 and Theorem 4.1

- See adaptive-flu.ipynb for examples to run synthetic sequential experiments on empirical data




Empirical data sets used in this paper are available at data/

- Data have been preprocessed into a matrix form

Figures are available in result/
