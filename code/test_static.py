import numpy as np
from utils_estimate import *
from utils_design import *
import time
from utils_empirical import *

def run_static(all_taus, seed=1234, print_epochs=100, all_Ys=None, pre_Ys=None, num_mc=100, N=100, T=50, pre_T=50, adj_pct=0.01, adjust_covar=False, J=2, G=2,
                     adjust_covar_only=False, method="OLS", return_std=False, lag=None, est_lag=None, no_bm=False, unit_effect=True, time_effect=True, all_names=None):
    np.random.seed(seed)

    if all_Ys is not None:
        num_mc = len(all_Ys)
        N, T = all_Ys[0].shape
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
    opt_treat_df = solve_static_opt_design(T, [0, lag])

    all_Zs = dict()
    if no_bm is False:
        for name in ['ff', 'ba', 'ffba']:
            all_Zs[name] = calc_cv_z_mtrx(N, T, bm_treat_df[name], cv=1)
        all_Zs['opt_0'] = calc_cv_z_mtrx(N, T, opt_treat_df[0], cv=1)
        if all_names is None:
            all_names = ['ff', 'ba', 'ffba', 'opt_0', 'opt']
    else:
        if all_names is None:
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
        if (j + 1) % print_epochs == 0:
            print("{}/{} done".format(j + 1, num_mc))

        this_ctrl_Y = all_Ys[j]

        if adjust_covar_only == False:
            for name in all_names:
                Z = all_Zs[name]
                Y = add_treatment_effect(this_ctrl_Y, Z, all_taus)
                if method == "GLS":
                    if return_std:
                        tau_hat, tau_var_hat = est_static_gls(Y, Z, est_lag, return_std=True)
                    else:
                        tau_hat = est_static_gls(Y, Z, est_lag)
                else:
                    if return_std:
                        tau_hat, tau_var_hat = est_static(Y, Z, est_lag, return_std=True, unit_effect=unit_effect, time_effect=time_effect)
                    else:
                        tau_hat = est_static(Y, Z, est_lag, unit_effect=unit_effect, time_effect=time_effect)
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
                    tau_hat, tau_var_hat = est_static_gls(Y, Z, est_lag, return_std=True)
                else:
                    tau_hat = est_static_gls(Y, Z, est_lag)
            else:
                if return_std:
                    tau_hat, tau_var_hat = est_static(Y, Z, est_lag, return_std=True, unit_effect=unit_effect, time_effect=time_effect)
                else:
                    tau_hat = est_static(Y, Z, est_lag, unit_effect=unit_effect, time_effect=time_effect)
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



def compare_static_designs(Y, all_taus, all_Ns, all_Ts, pre_T, num_mc=1000, num_iter=1, print_epochs = 100, adj_pct=0.02, method="GLS", J=1, G=4, seed=123, est_lag=None, all_names=None):
    timestamp = time.time()
    result = dict();
    result_all = dict()
    result_var = dict();
    result_all_var = dict()

    for N in all_Ns:
        for T in all_Ts:
            params = (N, np.sum(all_taus))
            np.random.seed(seed)
            all_idx_N_list = list();
            all_idx_T_list = list();
            all_shuffle_list = list()
            for _ in range(num_iter):
                idx_N_list, idx_T_list, shuffle_list = sample_flu_subblocks(Y, N, T, num_mc=num_mc, pre_T=pre_T)
                all_idx_N_list.append(idx_N_list);
                all_idx_T_list.append(idx_T_list);
                all_shuffle_list.append(shuffle_list)

            print(T)

            for j in range(num_iter):
                new_timestamp = time.time()
                print("iter: ", j, new_timestamp - timestamp)
                timestamp = new_timestamp

                idx_N_list = all_idx_N_list[j];
                idx_T_list = all_idx_T_list[j];
                shuffle_list = all_shuffle_list[j]
                pre_Ys, all_Ys = get_all_pre_Ys(Y, idx_N_list, idx_T_list, shuffle_list, pre_T)

                out_dict, out_total_dict, out_var_dict, out_total_var_dict = run_static(all_taus, all_Ys=all_Ys,
                                                                                        pre_Ys=pre_Ys,
                                                                                        print_epochs=print_epochs,
                                                                                        adjust_covar=False, return_std=True,
                                                                                        method=method, adj_pct=adj_pct,
                                                                                        est_lag=est_lag, all_names=all_names)

                this_dict, this_total_dict, this_var_dict, this_total_var_dict = run_static(all_taus, all_Ys=all_Ys,
                                                                                            pre_Ys=pre_Ys,
                                                                                            print_epochs=print_epochs,
                                                                                            J=J, G=G,
                                                                                            adjust_covar=True,
                                                                                            adjust_covar_only=True,
                                                                                            return_std=True,
                                                                                            method=method,
                                                                                            adj_pct=adj_pct, est_lag=est_lag)
                out_dict['opt+_' + str(J) + "_" + str(G)] = this_dict['opt+_' + str(J) + "_" + str(G)]
                out_total_dict['opt+_' + str(J) + "_" + str(G)] = this_total_dict['opt+_' + str(J) + "_" + str(G)]
                out_var_dict['opt+_' + str(J) + "_" + str(G)] = this_var_dict['opt+_' + str(J) + "_" + str(G)]
                out_total_var_dict['opt+_' + str(J) + "_" + str(G)] = this_total_var_dict[
                    'opt+_' + str(J) + "_" + str(G)]

                result[(N,T)] = out_dict
                result_all[(N,T)] = out_total_dict
                result_var[(N,T)] = out_var_dict
                result_all_var[(N,T)] = out_total_var_dict

    return result, result_all, result_var, result_all_var