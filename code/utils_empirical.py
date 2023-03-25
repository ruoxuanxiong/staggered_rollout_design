import os
import pickle
import pandas as pd
from utils_import_data import *
import seaborn as sns
import matplotlib.pyplot as plt


def save_results(results, params, save_path, replace=False):
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


def sample_flu_subblocks(data, N, T, pre_T=0, num_mc=500, interval=7):
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
    idx_N_list = [np.random.randint(data.shape[0] - N + 1) for _ in range(num_mc)]
    idx_T_list = [np.random.randint(data.shape[1] - T - pre_T + 1) for _ in range(num_mc)]
    shuffle_list = list()
    for j in range(num_mc):
        idx_N_list[j] = np.array(range(idx_N_list[j], idx_N_list[j]+N))
        idx_T_list[j] = np.array(range(idx_T_list[j], idx_T_list[j]+T+pre_T))
        shuffle_list.append(np.random.permutation(N))
    return idx_N_list, idx_T_list, shuffle_list


def get_all_Ys(full_Y, idx_N_list, idx_T_list, shuffle_list):
    all_Ys = list()
    num_mc = len(idx_N_list)
    for j in range(num_mc):
        all_Ys.append(full_Y[idx_N_list[j], :][:, idx_T_list[j]][shuffle_list[j], :])
    return all_Ys


def get_all_pre_Ys(full_Y, idx_N_list, idx_T_list, shuffle_list, T_pre):
    all_Ys = list(); pre_Ys = list()
    num_mc = len(idx_N_list)
    for j in range(num_mc):
        this_Y = full_Y[idx_N_list[j], :][:, idx_T_list[j]][shuffle_list[j], :]
        pre_Ys.append(this_Y[:,:T_pre]); all_Ys.append(this_Y[:,T_pre:])
    return pre_Ys, all_Ys


def calc_ma(Y, window):
    Y = pd.DataFrame(Y).T.rolling(window).mean()
    Y = Y.iloc[window:, :]
    Y = Y.T
    Y = Y.values
    return Y



def geval_color_palette():
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


def make_figure_vary_dimension(result, lag, all_taus, all_Ns, all_Ts, save_dir, J=1, G=4, scale=1, color_palette_dict=None):
    if color_palette_dict is None:
        color_palette_dict = geval_color_palette()
        
    if type(all_Ts) == list:
        varying_T = True
        val_col = '$T$'
        all_vals = all_Ts
        N = all_Ns
    else:
        varying_T = False
        val_col = '$N$'
        all_vals = all_Ns
        T = all_Ts
    out = dict()
    method_col = '$\mathrm{design}$'
    method_dict = {'ff': '$Z_{\mathrm{ff}}$', 'ba': '$Z_{\mathrm{ba}}$', 'ffba': '$Z_{\mathrm{ffba}}$',
                   'opt_0': '$Z_{\mathrm{opt,linear}}$', 'opt': '$Z_{\mathrm{opt}}$'}
    method_dict['opt+_' + str(J) + '_' + str(G)] = '$Z_{\mathrm{opt,stratified}}$'
    out[val_col] = list()
    out[method_col] = list()
    for idx in range(lag + 1):
        this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
        out[this_val_col] = list()

    for val in all_vals:
        if varying_T:
            val_key = (N, val)
        else:
            val_key = (val, T)
        for name in ['ff', 'ba', 'ffba', 'opt']:
            out[val_col] = out[val_col] + [val] * len(result[val_key][name])
            out[method_col] = out[method_col] + [method_dict[name]] * len(result[val_key][name])
            for idx in range(lag + 1):
                this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
                out[this_val_col] = out[this_val_col] + list(
                    (np.array(result[val_key][name])[:, idx] - all_taus[lag - idx]) ** 2)

    out_df = pd.DataFrame(out)
    total_val_col = '$\sum_{j}(\hat{\\tau}_{j}-\\tau_{j})^2$'
    out_df[total_val_col] = 0
    for idx in range(lag + 1):
        this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
        out_df[total_val_col] = out_df[total_val_col] + out_df[this_val_col]

    out_df.iloc[:, 2:] = out_df.iloc[:, 2:] * scale

    out_df_basic = out_df

    out[val_col] = list()
    out[method_col] = list()
    for idx in range(lag + 1):
        this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
        out[this_val_col] = list()

    for val in all_vals:
        if varying_T:
            val_key = (N, val)
        else:
            val_key = (val, T)
        for name in ['opt_0', 'opt']:
            out[val_col] = out[val_col] + [val] * len(result[val_key][name])
            out[method_col] = out[method_col] + [method_dict[name]] * len(result[val_key][name])
            for idx in range(lag + 1):
                this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
                out[this_val_col] = out[this_val_col] + list(
                    (np.array(result[val_key][name])[:, idx] - all_taus[lag - idx]) ** 2)

        name = 'opt+_' + str(J) + '_' + str(G)
        out[val_col] = out[val_col] + [val] * len(result[val_key][name])
        out[method_col] = out[method_col] + [method_dict[name]] * len(result[val_key][name])
        for idx in range(lag + 1):
            this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
            out[this_val_col] = out[this_val_col] + list(
                (np.array(result[val_key][name])[:, idx] - all_taus[lag - idx]) ** 2)

    out_df = pd.DataFrame(out)
    total_val_col = '$\sum_{j}(\hat{\\tau}_{j}-\\tau_{j})^2$'
    out_df[total_val_col] = 0
    for idx in range(lag + 1):
        this_val_col = '$(\hat{\\tau}_{\mathrm{' + str(idx) + '}}-\\tau_{\mathrm{' + str(idx) + '}})^2$'
        out_df[total_val_col] = out_df[total_val_col] + out_df[this_val_col]

    out_df.iloc[:, 2:] = out_df.iloc[:, 2:] * scale

    out_df_additional = out_df

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.lineplot(ax=axes[0], data=out_df_basic, x=val_col, y=total_val_col, hue='$\mathrm{design}$',
                 estimator=np.mean, err_style="band", palette=color_palette_dict)

    axes[0].tick_params('x', labelrotation=360)
    axes[0].set_xlabel(val_col)
    axes[0].set_ylabel(total_val_col)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid()

    sns.lineplot(ax=axes[1], data=out_df_additional, x=val_col, y=total_val_col, hue='$\mathrm{design}$',
                 estimator=np.mean, err_style="band", palette=color_palette_dict)

    axes[1].tick_params('x', labelrotation=360)
    axes[1].set_xlabel(val_col)
    axes[1].set_ylabel(total_val_col)

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid()
    plt.tight_layout()

    plt.savefig(save_dir+"_varying_T_lag_"+str(lag)+".pdf")


