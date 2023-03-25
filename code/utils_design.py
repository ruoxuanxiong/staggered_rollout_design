import pandas as pd
from utils_carryover import *
import scipy
from sklearn.cluster import KMeans

def generate_bm_design(T, adj_pct=0):
    """
    Generate the fraction of treated units per period for three benchmark designs (Z_ff, Z_ba, and Z_ffba)

    Parameters
    ----------
    T : experiment duration

    Returns
    -------
    A pandas dataframe with three columns. Each column is the treated fraction of one benchmark design
    """
    result = dict()

    result['ff'] = [0.5 - adj_pct if j < int(T/2) else 0.5 + adj_pct for j in range(T)]
    result['ba'] = [adj_pct] * int(T/2) + [1-adj_pct] * (T - int(T/2))
    result['ffba'] = [adj_pct] * int(T/2) + [0.5] * (T - int(T/2))


    out_df = pd.DataFrame(result)

    return out_df


def solve_nonadaptive_opt_design(T, all_lags):
    """
    Generate the fraction of treated units per period for nonlinear staggered design

    Parameters
    ----------
    T : experiment duration
    all_lags :  a list of all the possible ell (duration of carryover effects)

    Returns
    -------
    A pandas dataframe with multiple columns. Each column is the fraction of treated units per period when the duration of carryover effects equals a specific value in all_lags
    """
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


def add_treatment_effect(ctrl_Y, Z, all_taus):
    """
    Add the treatment effect to the control data

    Parameters
    ----------
    ctrl_Y : observed control data of dimension NxT
    Z : treatment design of dimension NxT
    all_taus : treatment effects [tau_0, \cdots, tau_\ell]

    Returns
    -------
    Synthetic experimental data
    """
    N, T = ctrl_Y.shape
    true_lag = len(all_taus) - 1
    padded_Z = np.append(np.zeros((N, true_lag)), Z + 1, axis=1)
    agg_Z = np.zeros((N, T))
    for l in range(true_lag + 1):
        agg_Z = agg_Z + padded_Z[:, (true_lag - l):(T + true_lag - l)] * all_taus[l]
    Y = ctrl_Y + agg_Z
    return Y


def find_opt_z_cluster(pre_Y, T, lag, J, G=2):
    """
    Stratify units based on the top singular vectors

    Parameters
    ----------
    pre_Y : historical control data
    T : experiment duration
    lag :  duration of carryover effects
    J : number of singular vector
    G : number of strata

    Returns
    -------
    A vector of stratum membership
    """
    N, pre_T = pre_Y.shape
    u, s, vh = np.linalg.svd(pre_Y, full_matrices=False)
    Uhat = u[:,:J]
    opt_treat_df = solve_nonadaptive_opt_design(T, [lag])

    kmeans = KMeans(n_clusters=G, random_state=0).fit(Uhat)
    labels = kmeans.labels_
    Z = np.zeros((N, T))

    for g in range(G):
        N_g = np.sum(labels == g)
        Z_g = calc_cv_z_mtrx(N_g, T, opt_treat_df[lag], cv=1)
        Z[labels == g, :] = Z_g
    return Z


def calc_cv_z_mtrx(N, T, treat_avg, cv=2):
    """
    Generate the treatment design matrix Z based on the fraction of treated units per period

    Parameters
    ----------
    N : number of experimental units
    T : experiment duration
    treat_avg :  fraction of treated units per period
    cv : number of equally sized subsets of units, where treated fraction of each subset satisfies treat_avg

    Returns
    -------
    An NxT treatment design matrix
    """
    sub_N = int(N/cv)
    st_N_treat = [int(round(treat_avg[j] * sub_N)) for j in range(T)]
    zs = -np.ones((N, T));
    for t in range(T):
        for s in range(cv):
            zs[(sub_N*(s+1) - st_N_treat[t]):(sub_N*(s+1)), t] = 1
    return zs