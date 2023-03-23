import numpy as np
import pandas as pd
import scipy
from utils_carryover import theta1, theta2
from utils import calc_cv_z_mtrx, generate_bm_design, within_transform
from sklearn.cluster import KMeans



def solve_static_opt_design(T, all_lags):
    result = dict();
    result[0] = [(2 * t + 1) / (2 * T) for t in range(T)]
    for lag in all_lags:
        def crossover_fun(y, L=lag):
            H = theta1(y, L=L) - theta2(y, L=L)
            return -np.sum(np.diag(H))

        y0 = [((2 * t + 1) / T - 1) for t in range(T)]
        bnds = tuple([(-1, 1) for t in range(T)])
        cons = tuple([{'type': 'ineq', 'fun': lambda y: y[t] - y[t - 1]} for t in range(1, T)])
        res = scipy.optimize.minimize(crossover_fun, y0, constraints=cons, bounds=bnds)
        result[lag] = (1 + res.x) / 2
    out_static_df = pd.DataFrame(result)
    return out_static_df




def find_opt_z_cluster(pre_Y, T, lag, J, G=2):

    N, pre_T = pre_Y.shape
    u, s, vh = np.linalg.svd(pre_Y, full_matrices=False)
    Uhat = u[:,:J]
    opt_treat_df = solve_static_opt_design(T, [lag])

    kmeans = KMeans(n_clusters=G, random_state=0).fit(Uhat)
    labels = kmeans.labels_
    Z = np.zeros((N, T))

    for g in range(G):
        N_g = np.sum(labels == g)
        Z_g = calc_cv_z_mtrx(N_g, T, opt_treat_df[lag], cv=1)
        Z[labels == g, :] = Z_g
    return Z


def est_static(Y, Z, lag, return_std=False, return_resid=False, unit_effect=True, time_effect=True):
    N, T = Y.shape
    Y_wi = within_transform(Y[:, lag:], unit_effect=unit_effect, time_effect=time_effect)
    all_Z_wi = list()
    for l in range(lag + 1):
        all_Z_wi.append(within_transform(Z[:, l:(T - lag + l)], unit_effect=unit_effect, time_effect=time_effect))

    y_vec = Y_wi.T.reshape((N * (T - lag), 1))
    z_matrix = np.zeros((N * (T - lag), lag + 1))
    for l in range(lag + 1):
        z_matrix[:, l] = all_Z_wi[l].T.reshape((N * (T - lag),))

    X = z_matrix
    coef = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y_vec))
    resid = y_vec - X.dot(coef)
    resid = resid.reshape(((T - lag), N)).T

    avg_resid_sq = np.mean(resid ** 2)
    var = avg_resid_sq * np.linalg.inv(X.T.dot(X))

    if return_std:
        if return_resid:
            return list(coef[:, 0]), list(np.diag(var)), resid
        else:
            return list(coef[:, 0]), list(np.diag(var))
    else:
        if return_resid:
            return list(coef[:, 0]), resid
        else:
            return list(coef[:, 0])


def est_static_gls(Y, Z, lag, return_std=False, return_resid=False, rank=1):
    N, T = Y.shape
    coef, resid = est_static(Y, Z, lag, return_resid=True)
    resid_cross_cov = resid.dot(resid.T) / (T - lag + 1)

    U, S, Vh = np.linalg.svd(resid_cross_cov)
    M1 = U[:, :rank].dot(np.diag(S[:rank])).dot(Vh[:rank, :])
    resid_cov_adj = M1 + np.diag(np.diag(resid_cross_cov - M1))

    resid_cov_adj_inv = np.linalg.inv(resid_cov_adj)
    inv_resid_cov = np.zeros((N * (T - lag), N * (T - lag)))
    for t in range(T - lag):
        inv_resid_cov[(N * t):(N * (t + 1)), (N * t):(N * (t + 1))] = resid_cov_adj_inv

    I_tilde = np.append(np.diag(np.ones((N - 1,))), np.zeros((1, N - 1)), axis=0)
    one_vec = np.ones((N, 1))
    Gamma = np.zeros((N * (T - lag), N + T - lag - 1))
    for t in range(T - lag):
        Gamma[(N * t):(N * (t + 1)), :(N - 1)] = I_tilde
        Gamma[(N * t):(N * (t + 1)), (N - 1 + t):(N + t)] = one_vec

    inv_resid_cov_adj = inv_resid_cov - inv_resid_cov.dot(Gamma).dot(
        np.linalg.inv(Gamma.T.dot(inv_resid_cov).dot(Gamma))).dot(Gamma.T).dot(inv_resid_cov)

    y_vec = Y[:, lag:].T.reshape((N * (T - lag), 1))
    z_matrix = np.zeros((N * (T - lag), lag + 1))
    for l in range(lag + 1):
        z_matrix[:, l] = Z[:, l:(T - lag + l)].T.reshape((N * (T - lag),))
    coef = np.linalg.inv(z_matrix.T.dot(inv_resid_cov_adj).dot(z_matrix)).dot(
        z_matrix.T.dot(inv_resid_cov_adj).dot(y_vec))


    if return_std | return_resid:
        Y_wi = within_transform(Y[:, lag:])
        all_Z_wi = list()
        for l in range(lag + 1):
            all_Z_wi.append(within_transform(Z[:, l:(T - lag + l)]))

        y_vec = Y_wi.T.reshape((N * (T - lag), 1))
        z_matrix = np.zeros((N * (T - lag), lag + 1))
        for l in range(lag + 1):
            z_matrix[:, l] = all_Z_wi[l].T.reshape((N * (T - lag),))
        resid = y_vec - z_matrix.dot(coef)
        resid = resid.reshape(((T - lag), N)).T

        # avg_resid_sq = np.mean(resid ** 2)
        # print(avg_resid_sq)

        resid_cross_cov = resid.dot(resid.T) / (T - lag + 1)

        U, S, Vh = np.linalg.svd(resid_cross_cov)
        M1 = U[:, :rank].dot(np.diag(S[:rank])).dot(Vh[:rank, :])
        resid_cov_adj = M1 + np.diag(np.diag(resid_cross_cov - M1))
        resid_cov = np.zeros((N * (T - lag), N * (T - lag)))
        for t in range(T - lag):
            resid_cov[(N * t):(N * (t + 1)), (N * t):(N * (t + 1))] = resid_cov_adj

        left_M = np.linalg.inv(z_matrix.T.dot(inv_resid_cov_adj).dot(z_matrix))
        middle_M_1 = z_matrix.T.dot(inv_resid_cov_adj)
        var = left_M.dot(middle_M_1.dot(resid_cov).dot(middle_M_1.T)).dot(left_M)

    if return_std:
        if return_resid:
            return list(coef[:, 0]), list(np.diag(var)), resid
        else:
            return list(coef[:, 0]), list(np.diag(var))
    else:
        if return_resid:
            return list(coef[:, 0]), resid
        else:
            return list(coef[:, 0])


def gls(Y, Z, all_tau, return_std=True, method="GLS", rank=None, global_est=False):
    lag = len(all_tau)
    N, T = Y.shape
    for l in range(lag):
        Y[:, l:] = Y[:, l:] + (1 + Z[:, :(T - l)]) * all_tau[l]

    I_tilde = np.append(np.diag(np.ones((N - 1,))), np.zeros((1, N - 1)), axis=0)
    one_vec = np.ones((N, 1))
    Gamma = np.zeros((N * (T - lag + 1), N + T - lag))
    for t in range(T - lag + 1):
        Gamma[(N * t):(N * (t + 1)), :(N - 1)] = I_tilde
        Gamma[(N * t):(N * (t + 1)), (N - 1 + t):(N + t)] = one_vec

    y_vec = Y[:, (lag - 1):].T.reshape((N * (T - lag + 1), 1))
    z_matrix = np.zeros((N * (T - lag + 1), lag))
    for l in range(lag):
        z_matrix[:, l] = Z[:, l:(T - lag + 1 + l)].T.reshape((N * (T - lag + 1),))
    # z_vec = Z.T.reshape((N * T, 1))
    # X = np.append(z_vec, Gamma, axis=1)
    X = np.append(z_matrix, Gamma, axis=1)


    coef = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y_vec))
    resid = y_vec - X.dot(coef)
    resid_cov = resid.dot(resid.T)
    if method == "GLS":
        if rank is None:
            rank = 1 # set rank as 1
        if global_est:
            U, S, Vh = np.linalg.svd(resid_cov, full_matrix=False)
            if rank > 0:
                M1 = U[:, :rank].dot(np.diag(S[:rank])).dot(Vh[:rank, :])
            else:
                M1 = np.zeros((N * (T - lag + 1), N * (T - lag + 1)))
            resid_cov_adj = M1 + np.diag(np.diag(resid_cov - M1))
            inv_resid_cov = np.linalg.inv(resid_cov_adj)
        else:
            resid_mtrx = resid.reshape((T - lag + 1, N)).T
            resid_cross_cov = resid_mtrx.dot(resid_mtrx.T) / T

            U, S, Vh = np.linalg.svd(resid_cross_cov)
            if rank > 0:
                M1 = U[:, :rank].dot(np.diag(S[:rank])).dot(Vh[:rank, :])
            else:
                M1 = np.zeros((N, N))
            resid_cov_adj = M1 + np.diag(np.diag(resid_cross_cov - M1))
            resid_cov_adj_inv = np.linalg.inv(resid_cov_adj)
            inv_resid_cov = np.zeros((N * (T - lag + 1), N * (T - lag + 1)))
            for t in range(T - lag + 1):
                inv_resid_cov[(N * t):(N * (t + 1)), (N * t):(N * (t + 1))] = resid_cov_adj_inv
    else:
        inv_resid_cov = np.diag(1 / np.diag(resid_cov))
    coef = np.linalg.inv(X.T.dot(inv_resid_cov).dot(X)).dot(X.T.dot(inv_resid_cov).dot(y_vec))


    if return_std:
        resid = y_vec - X.dot(coef)
        resid = resid.reshape(((T - lag + 1), N)).T

        resid_cross_cov = resid.dot(resid.T) / T

        U, S, Vh = np.linalg.svd(resid_cross_cov)
        if rank > 0:
            M1 = U[:, :rank].dot(np.diag(S[:rank])).dot(Vh[:rank, :])
        else:
            M1 = np.zeros((N, N))
        resid_cov_adj = M1 + np.diag(np.diag(resid_cross_cov - M1))
        resid_cov = np.zeros((N * (T - lag + 1), N * (T - lag + 1)))
        for t in range(T - lag + 1):
            resid_cov[(N * t):(N * (t + 1)), (N * t):(N * (t + 1))] = resid_cov_adj

        var = np.linalg.inv(X.T.dot(inv_resid_cov).dot(X)).dot(
            X.T.dot(inv_resid_cov).dot(resid_cov).dot(inv_resid_cov).dot(X)).dot(
            np.linalg.inv(X.T.dot(inv_resid_cov).dot(X)))

        return list(coef[:lag, 0]), list(np.diag(var[:lag, :lag]))
    else:
        return list(coef[:lag, 0])


def add_treatment_effect(ctrl_Y, Z, all_taus):
    N, T = ctrl_Y.shape
    true_lag = len(all_taus) - 1
    padded_Z = np.append(np.zeros((N, true_lag)), Z + 1, axis=1)
    agg_Z = np.zeros((N, T))
    for l in range(true_lag + 1):
        agg_Z = agg_Z + padded_Z[:, (true_lag - l):(T + true_lag - l)] * all_taus[l]
    Y = ctrl_Y + agg_Z
    return Y


def static_covariate(all_taus, seed=1234, print_epochs=100, all_Ys=None, pre_Ys=None, num_mc=100, N=100, T=50, pre_T=50, adj_pct=0.01, adjust_covar=False, J=2, G=2,
                     adjust_covar_only=False, method="OLS", return_std=False, lag=None, est_lag=None, no_bm=False, unit_effect=True, time_effect=True):
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
    opt_treat_df = solve_static_opt_design(T, [0, lag])

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