import os
import pickle
import pandas as pd
from utils_import_data import *
import seaborn as sns

def sample_flu_subblocks(data, N, T, pre_T=0, num_mc=500, interval=7):
    """
    generate m lists of N indices of units and m lists of T indices of time periods for the flu data

    Parameters
    ----------
    data : full panel of observed data
    N: number of units
    T: experiment duration
    pre_T: number of time periods in the historical data
    num_mc: m lists
    interval : restriction on the initial time period of the experiment (beginning of the flu season)

    Returns
    -------
    A list of m lists of indices of units, a list of m lists of indices of time periods, and a list of m permutations of units
    """
    idx_data_T = int(data.shape[1]/interval); idx_pre_T = int(pre_T/interval); idx_T = int(T/interval)
    idx_N_list = [np.random.randint(data.shape[0] - N + 1) for _ in range(num_mc)]
    idx_T_list = [np.random.randint(idx_data_T-idx_T-idx_pre_T)*interval for _ in range(num_mc)]
    shuffle_list = list()
    for j in range(num_mc):
        idx_N_list[j] = np.array(range(idx_N_list[j], idx_N_list[j]+N))
        idx_T_list[j] = np.array(range(idx_T_list[j], idx_T_list[j]+T+pre_T))
        shuffle_list.append(np.random.permutation(N))
    return idx_N_list, idx_T_list, shuffle_list


def sample_subblocks(data, N, T, pre_T=0, num_mc=500):
    """
    generate m lists of N indices of units and m lists of T indices of time periods

    Parameters
    ----------
    data : full panel of observed data
    N: number of units
    T: experiment duration
    pre_T: number of time periods in the historical data
    num_mc: m lists

    Returns
    -------
    A list of m lists of indices of units, a list of m lists of indices of time periods, and a list of m permutations of units
    """
    idx_N_list = [np.random.randint(data.shape[0] - N + 1) for _ in range(num_mc)]
    idx_T_list = [np.random.randint(data.shape[1] - T - pre_T + 1) for _ in range(num_mc)]
    shuffle_list = list()
    for j in range(num_mc):
        idx_N_list[j] = np.array(range(idx_N_list[j], idx_N_list[j]+N))
        idx_T_list[j] = np.array(range(idx_T_list[j], idx_T_list[j]+T+pre_T))
        shuffle_list.append(np.random.permutation(N))
    return idx_N_list, idx_T_list, shuffle_list


def get_all_Ys(full_Y, idx_N_list, idx_T_list, shuffle_list):
    """
    extract m subblocks of observed control data based on the indices of the units and time periods

    Parameters
    ----------
    full_Y : full panel of observed data
    idx_N_list: list of m lists of indices of units
    idx_T_list: list of m lists of indices of time periods
    shuffle_list: list of m permutations of units

    Returns
    -------
    A list of m observed control data
    """
    all_Ys = list()
    num_mc = len(idx_N_list)
    for j in range(num_mc):
        all_Ys.append(full_Y[idx_N_list[j], :][:, idx_T_list[j]][shuffle_list[j], :])
    return all_Ys


def get_all_pre_Ys(full_Y, idx_N_list, idx_T_list, shuffle_list, T_pre):
    """
    extract m subblocks of hostorical and observed control data based on the indices of the units and time periods

    Parameters
    ----------
    full_Y : full panel of observed data
    idx_N_list: list of m lists of indices of units
    idx_T_list: list of m lists of indices of time periods
    shuffle_list: list of m permutations of units
    T_pre: number of time periods in the historical data

    Returns
    -------
    A list of m hostorical data sets and a list of m observed control data
    """
    all_Ys = list(); pre_Ys = list()
    num_mc = len(idx_N_list)
    for j in range(num_mc):
        this_Y = full_Y[idx_N_list[j], :][:, idx_T_list[j]][shuffle_list[j], :]
        pre_Ys.append(this_Y[:,:T_pre]); all_Ys.append(this_Y[:,T_pre:])
    return pre_Ys, all_Ys


def calc_ma(Y, window):
    """
    Smooth the data

    Parameters
    ----------
    Y : matrix to be smoothed

    Returns
    -------
    Smoothed matrix
    """
    Y = pd.DataFrame(Y).T.rolling(window).mean()
    Y = Y.iloc[window:, :]
    Y = Y.T
    Y = Y.values
    return Y



def geval_color_palette():
    """
    A color palette for various treatment designs

    Returns
    -------
    A dictionary that specifies the color for each treatment design
    """
    color_palette = sns.color_palette()
    color_palette_dict = dict()
    color_palette_dict['$Z_{\mathrm{ff}}$'] = color_palette[0]
    color_palette_dict['$Z_{\mathrm{ba}}$'] = color_palette[1]
    color_palette_dict['$Z_{\mathrm{ffba}}$'] = color_palette[2]
    color_palette_dict['$Z_{\mathrm{opt}}$'] = color_palette[3]
    color_palette_dict['$Z_{\mathrm{opt,nonlinear}}$'] = color_palette[3]

    color_palette_dict['$Z_{\mathrm{opt,linear}}$'] = color_palette[4]

    color_palette_dict['$Z_{\mathrm{opt,stratified}}$'] = color_palette[5]
    for G in range(2, 5):
        color_palette_dict['$Z_{\mathrm{opt,}K=' + str(G) + '}$'] = color_palette[3 + G]

    return color_palette_dict


def save_results(results, params, save_path, replace=False):
    """
    Save results to pickle

    Parameters
    ----------
    results : results to be saved
    params : key in the dictionary of the results
    save_path :  path the save the results
    """

    if not os.path.exists(save_path):
        summary = dict()
        summary[params] = results
    else:
        if replace:
            summary = dict()
        else:
            with open(save_path, 'rb') as handle:
                summary = pickle.load(handle)
        summary[params] = results

    with open(save_path, 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)