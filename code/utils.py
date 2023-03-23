import numpy as np
import pandas as pd


def within_transform(Y, unit_effect=True, time_effect=True):
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

def generate_bm_design(T, adj_pct=0):
    result = dict()

    result['ff'] = [0.5 - adj_pct if j < int(T/2) else 0.5 + adj_pct for j in range(T)]
    result['ba'] = [adj_pct] * int(T/2) + [1-adj_pct] * (T - int(T/2))
    result['ffba'] = [adj_pct] * int(T/2) + [0.5] * (T - int(T/2))


    out_df = pd.DataFrame(result)

    return out_df


def calc_cv_z_mtrx(N, T, treat_avg, cv=2):
    sub_N = int(N/cv)
    st_N_treat = [int(round(treat_avg[j] * sub_N)) for j in range(T)]
    zs = -np.ones((N, T));
    for t in range(T):
        for s in range(cv):
            zs[(sub_N*(s+1) - st_N_treat[t]):(sub_N*(s+1)), t] = 1
        # zs[(N - st_N_treat[t]):N, t] = 1
    return zs

