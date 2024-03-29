import numpy as np

def within_transform(Y, unit_effect=True, time_effect=True):
    """
    Within transformation of a matrix

    Parameters
    ----------
    Y : matrix to be transformed
    unit_effect : whether to subtract away the mean of a unit's time series observations in the within transformation
    time_effect :  whether to subtract away the mean of the observations of all units at a specific time period in the within transformation

    Returns
    -------
    Within transformed matrix
    """
    N, T = Y.shape
    row_avg = np.mean(Y, axis=1).reshape((N,1))
    col_avg = np.mean(Y,axis=0).reshape((T,1))
    total_avg = np.mean(Y)
    if unit_effect and time_effect:
        Y_wi = Y - row_avg.dot(np.ones((1,T))) - np.ones((N,1)).dot(col_avg.T) + total_avg * np.ones((N,T))
    elif (unit_effect is False) and time_effect:
        Y_wi = Y - np.ones((N, 1)).dot(col_avg.T)
    elif unit_effect and (time_effect is False):
        Y_wi = Y - row_avg.dot(np.ones((1,T)))
    else:
        Y_wi = Y.copy()
    return Y_wi


def est_ols(Y, Z, lag, return_std=False, return_resid=False, unit_effect=True, time_effect=True):
    """
    ordinary least squares estimator

    Parameters
    ----------
    Y : observed outcome matrix
    Z : treatment design matrix
    lag : duration of carryover effects ell
    return_std : whether to return variance-covariance matrix of the estimated treatment effects
    return_resid : whether to return the residuals
    unit_effect : whether the outcome specification has unit fixed effect
    time_effect : whether the outcome specification has time fixed effect

    Returns
    -------
    estimated tratment effects, variance-covariance matrix of the estimated treatment effects (optional), residuals (optional)

    """
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



def est_gls(Y, Z, lag, return_std=False, return_resid=False, rank=1):
    """

    generalized least squares estimator

    Parameters
    ----------
    Y : observed outcome matrix
    Z : treatment design matrix
    lag : duration of carryover effects ell
    return_std : whether to return variance-covariance matrix of the estimated treatment effects
    return_resid : whether to return the residuals
    rank: number of latent covariates (dimension of u_i)

    Returns
    -------
    estimated tratment effects, variance-covariance matrix of the estimated treatment effects (optional), residuals (optional)

    """
    N, T = Y.shape
    coef, resid = est_ols(Y, Z, lag, return_resid=True)
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




