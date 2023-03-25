import numpy as np
from utils_estimate import est_ols, est_gls
from utils_design import *



def run_nonadaptive(all_taus, seed=1234, print_epochs=100, all_Ys=None, pre_Ys=None, num_mc=100, N=100, T=50, pre_T=50, adj_pct=0.01, adjust_covar=False, J=2, G=2,
                     adjust_covar_only=False, method="OLS", return_std=False, lag=None, est_lag=None, no_bm=False, unit_effect=True, time_effect=True):
    """

    Run synthetic nonadaptive experiments

    Parameters
    ----------
    all_taus : true treatment effects
    seed : random seed
    print_epochs : number of epochs to print progress
    all_Ys : list of m subblocks of observed control data
    pre_Ys : list of m subblocks of historical control data
    num_mc : number of terations
    N : number of experimental units (need to specify if all_Ys is None)
    T : number of experiment duration (need to specify if all_Ys is None)
    pre_T : number of time periods in the historical control data (need to specify if pre_Ys is None)
    adj_pct : fraction of adjustment units in the benchmark designs to ensure that singularity problem does not arise in the estimation of treatment effects
    adjust_covar : whether to stratify in the design matrix
    J : number of singular vectors used for stratification
    G : number of strata
    adjust_covar_only : whether to run synthetic experiment only using stratified design (adjust_covar_only=True) or not (adjust_covar_only=False)
    method : estimation method for treatment effects (either OLS or GLS)
    return_std : whether to return variance-covariance matrix of the estimated treatment effects
    lag : true duration of carryover effects
    est_lag : duration of carryover effects specified in the estimation of treatment effects
    no_bm : whether to run synthetic experiments using benchmark designs (no_bm = False) or not (no_bm = True)
    unit_effect : whether the outcome specification has unit fixed effect
    time_effect : whether the outcome specification has time fixed effect

    Returns
    -------
    A list of estimated treatment effects, a list of estimated total treatment effects (optional),
    a list of variance-covariance matrix of the estimated treatment effects (optional), a list of variance-covariance matrix of the estimated total treatment effects (optional)

    """
    np.random.seed(seed)

    if all_Ys is not None:
        num_mc = len(all_Ys)
        N, T = all_Ys[0].shape
        if adjust_covar:
            pre_T = pre_Ys[0].shape[1]
    else:
        all_Ys = list()
        for j in range(num_mc):
            this_Y = np.random.normal(size=(N, 1)).dot(np.ones((1,T))) + np.ones((N,1)).dot(np.random.normal(size=(1, T))) + np.random.normal(size=(N, T))
            all_Ys.append(this_Y)

        if adjust_covar:
            pre_Ys = list()
            for j in range(num_mc):
                pre_Y = np.random.normal(size=(N, 1)).dot(np.ones((1, pre_T))) + np.ones((N, 1)).dot(
                    np.random.normal(size=(1, pre_T))) + np.random.normal(size=(N, pre_T))
                pre_Ys.append(pre_Y)


    if lag is None:
        lag = len(all_taus) - 1

    if est_lag is None:
        est_lag = len(all_taus) - 1
    bm_treat_df = generate_bm_design(T, adj_pct=adj_pct)
    opt_treat_df = solve_nonadaptive_opt_design(T, [0, lag])

    all_Zs = dict()
    if no_bm is False:
        for name in ['ff', 'ba', 'ffba']:
            all_Zs[name] = calc_cv_z_mtrx(N, T, bm_treat_df[name], cv=1)
        all_Zs['opt_0'] = calc_cv_z_mtrx(N, T, opt_treat_df[0], cv=1)
        all_names = ['ff', 'ba', 'ffba', 'opt_0', 'opt']
    else:
        all_names = ['opt']


    all_Zs['opt'] = calc_cv_z_mtrx(N, T, opt_treat_df[lag], cv=1)

    out_dict = dict(); out_total_dict = dict()

    if return_std:
        out_var_dict = dict()
        out_total_var_dict = dict()

    if adjust_covar_only == False:
        for name in all_names:
            out_dict[name] = list(); out_total_dict[name] = list()
            if return_std:
                out_var_dict[name] = list();
                out_total_var_dict[name] = list()

    if adjust_covar:
        out_dict['opt+_' + str(J) + "_" + str(G)] = list()
        out_total_dict['opt+_' + str(J) + "_" + str(G)] = list()
        if return_std:
            out_var_dict['opt+_' + str(J) + "_" + str(G)] = list()
            out_total_var_dict['opt+_' + str(J) + "_" + str(G)] = list()

    for j in range(num_mc):
        # print(j)
        if (j + 1) % print_epochs == 0:
            print("{}/{} done".format(j + 1, num_mc))

        this_ctrl_Y = all_Ys[j]

        if adjust_covar_only == False:
            for name in all_names:
                Z = all_Zs[name]
                Y = add_treatment_effect(this_ctrl_Y, Z, all_taus)
                if method == "GLS":
                    if return_std:
                        tau_hat, tau_var_hat = est_gls(Y, Z, est_lag, return_std=True)
                    else:
                        tau_hat = est_gls(Y, Z, est_lag)
                else:
                    if return_std:
                        tau_hat, tau_var_hat = est_ols(Y, Z, est_lag, return_std=True, unit_effect=unit_effect, time_effect=time_effect)
                    else:
                        tau_hat = est_ols(Y, Z, est_lag, unit_effect=unit_effect, time_effect=time_effect)
                tau_hat_total = np.sum(tau_hat)
                out_dict[name].append(tau_hat)
                out_total_dict[name].append(tau_hat_total)

                if return_std:
                    tau_var_total = np.sum(tau_var_hat)
                    out_var_dict[name].append(tau_var_hat)
                    out_total_var_dict[name].append(tau_var_total)

        if adjust_covar:
            this_pre_Y = pre_Ys[j]
            Z = find_opt_z_cluster(this_pre_Y, T, lag, J=J, G=G)
            Y = add_treatment_effect(this_ctrl_Y, Z, all_taus)
            if method == "GLS":
                if return_std:
                    tau_hat, tau_var_hat = est_gls(Y, Z, est_lag, return_std=True)
                else:
                    tau_hat = est_gls(Y, Z, est_lag)
            else:
                if return_std:
                    tau_hat, tau_var_hat = est_ols(Y, Z, est_lag, return_std=True, unit_effect=unit_effect, time_effect=time_effect)
                else:
                    tau_hat = est_ols(Y, Z, est_lag, unit_effect=unit_effect, time_effect=time_effect)
            tau_hat_total = np.sum(tau_hat)
            out_dict['opt+_'+str(J)+"_"+str(G)].append(tau_hat)
            out_total_dict['opt+_'+str(J)+"_"+str(G)].append(tau_hat_total)

            if return_std:
                tau_var_total = np.sum(tau_var_hat)
                out_var_dict['opt+_'+str(J)+"_"+str(G)].append(tau_var_hat)
                out_total_var_dict['opt+_'+str(J)+"_"+str(G)].append(tau_var_total)



    if return_std:
        return out_dict, out_total_dict, out_var_dict, out_total_var_dict
    else:
        return out_dict, out_total_dict