import numpy as np
import scipy
from utils_design import calc_cv_z_mtrx
from utils_estimate import within_transform
from utils_carryover import theta1, theta2

def est_within(Y, Z):
    N, T = Y.shape
    Y_wi = within_transform(Y)
    Z_wi = within_transform(Z)
    y_vec = Y_wi.T.reshape((N*T,1))
    z_vec = Z_wi.T.reshape((N*T,1))
    coef = np.linalg.inv(z_vec.T.dot(z_vec)).dot(z_vec.T.dot(y_vec))
    resid = y_vec - z_vec.dot(coef)
    resid = resid.reshape((T,N)).T
    avg_resid_sq = np.mean(resid ** 2)
    var = avg_resid_sq * np.linalg.inv(z_vec.T.dot(z_vec))
    return coef[0,0], var[0,0], resid


def calc_Phi(Z):
    Z_wi = within_transform(Z)
    Phi = np.mean(Z_wi ** 2)
    return Phi


def calc_sigma_sq(resid):
    # resid is an NxT matrix
    N, T = resid.shape
    sigma_sq_hat = np.sum(resid ** 2) / (N * (T - 1))
    return sigma_sq_hat


def calc_xi_sq(resid):
    N, T = resid.shape
    resid_sq = resid ** 2
    sigma_sq_hat = calc_sigma_sq(resid)
    xi_sq_hat = np.mean(((resid_sq - sigma_sq_hat).dot(np.ones((T, 1)))) ** 2) * T / (
            (T - 1) ** 2) - (
                        3 * T - 2) / ((T - 1) ** 2) * (sigma_sq_hat ** 2)

    return xi_sq_hat


def calc_tau_hat_var(Z, resid):
    Phi = calc_Phi(Z)
    Phi_inv = 1/Phi
    sigma_sq_hat = calc_sigma_sq(resid)
    return sigma_sq_hat * Phi_inv


def calc_sigma_sq_hat_var(resid):
    N, T = resid.shape
    sigma_sq_hat = calc_sigma_sq(resid)
    xi_sq_hat = calc_xi_sq(resid)
    xi_dagger_sq_hat = xi_sq_hat + 2 * sigma_sq_hat ** 2 / (T - 1)
    return xi_dagger_sq_hat


def calc_scaled_Phi_list(bm_treat_avg, T_max, N):
    # scaled_Phi is Phi multipled by T
    Z = calc_cv_z_mtrx(N, T_max, bm_treat_avg, cv=1)
    scaled_Phi_list = list()
    for t in range(1,T_max+1):
        scaled_Phi_list.append(calc_Phi(Z[:,:t]) * t)

    scaled_Phi_list[0] = scaled_Phi_list[1]/2 ## otherwise const_list[0] = 0
    return scaled_Phi_list


def get_Pt(sigma_sq, scaled_xi_dagger_sq, N, t, scaled_Phi_list, prec_thres, num_mc=100):
    xi_dagger_sq_sqrt = scaled_xi_dagger_sq ** 0.5
    Pt = np.zeros((len(scaled_Phi_list),))
    for j in range(num_mc):
        this_sigma_sq = np.random.normal(loc=sigma_sq, scale=xi_dagger_sq_sqrt) # 0724 update
        this_prec_list = N * np.array(scaled_Phi_list) / this_sigma_sq
        # how many prec is less than the target
        min_T = np.sum(this_prec_list < prec_thres)


        if min_T == len(scaled_Phi_list):
            min_T = len(scaled_Phi_list) - 1

        if min_T < t:
            min_T = t
        Pt[min_T] = Pt[min_T] + 1

    Pt = Pt / np.sum(Pt)

    return Pt


def solve_nonadaptive_opt_design(T, fix_y=[]):
    fix_y_len=len(fix_y)
    if fix_y_len > 0:
        fix_y_bnds = [(fix_y[t], fix_y[t]) for t in range(fix_y_len)]
    else:
        fix_y_bnds = []
    result = dict();
    result[0] = [(2 * t + 1) / (2 * T) for t in range(T)]

    def carryover_fun(y):
        H = theta1(y, L=0) - theta2(y, L=0)
        return -np.sum(np.diag(H))

    y0 = [((2 * t + 1) / T - 1) for t in range(T)]
    bnds = tuple(fix_y_bnds + [(-1, 1) for t in range(fix_y_len, T)])
    cons = tuple([{'type': 'ineq', 'fun': lambda y: y[t] - y[t - 1]} for t in range(1, T)])
    res = scipy.optimize.minimize(carryover_fun, y0, constraints=cons, bounds=bnds)
    return res


def dp_next_w(ad_w_t, Pt, T_max, t, N0=50, scale=10e8):
    all_ws = [(t - N0/2) / (N0/2) for t in range(0, N0+1)]
    min_val = 100;
    opt_w = ad_w_t[-1]
    for w in all_ws:

        if w >= ad_w_t[-1]:
            this_val = 0
            for T in range(t + 1, T_max):
                if Pt[T - 1] > 0:
                    res = solve_nonadaptive_opt_design(T, fix_y=ad_w_t + [w])
                    val = round(scale * 1 / (-res.fun) * T) / scale
                    this_val = this_val + val * Pt[T - 1]
            if this_val < min_val:
                min_val = this_val
                opt_w = w

    return opt_w


def test_asymptotics(N, T_max, tau, T=None, seed=1234, num_mc=1000, sigma=1):
    np.random.seed(seed)
    bm_treat_avg = [(2 * t + 1) / (2 * T_max) for t in range(T_max)]
    Z = calc_cv_z_mtrx(N, T_max, bm_treat_avg, cv=1)
    Y_avg = np.random.normal(size=(N, 1)).dot(np.ones((1, T_max))) + np.ones((N, 1)).dot(
        np.random.normal(size=(1, T_max)))

    if T is None:
        T = T_max

    out = {'tau_err': [], 'sigma_err': [], 'tau_hat': [], 'sigma_sq_hat': [], 'xi_sq_hat': [],
           'tau_err_std': [], 'sigma_err_std': [], 'sigma_err_std_wrong': []}

    for j in range(num_mc):
        e = np.random.normal(size=(N, T_max)) * sigma
        Y = Y_avg + e
        Y = Y + (1+Z) * tau

        tau_hat, tau_hat_var, eps_hat = est_within(Y[:,:T], Z[:,:T])
        sigma_sq_hat = calc_sigma_sq(eps_hat)
        xi_sq_hat = calc_xi_sq(eps_hat)

        tau_hat_var = calc_tau_hat_var(Z, eps_hat)
        xi_dagger_sq_hat = calc_sigma_sq_hat_var(eps_hat)

        eps_sq_hat_mat = eps_hat ** 2
        xi_sq_hat_2 = np.mean((eps_sq_hat_mat - sigma_sq_hat)**2)

        out['tau_hat'].append(tau_hat)
        out['tau_err'].append(tau_hat - tau)

        out['sigma_sq_hat'].append(sigma_sq_hat)
        out['sigma_err'].append(sigma_sq_hat - sigma**2)

        out['xi_sq_hat'].append(xi_sq_hat)

        out['tau_err_std'].append((tau_hat - tau) / np.sqrt(tau_hat_var) * np.sqrt(N * T))
        out['sigma_err_std'].append((sigma_sq_hat - sigma ** 2) / np.sqrt(xi_dagger_sq_hat) * np.sqrt(N * T))

        out['sigma_err_std_wrong'].append(
            (sigma_sq_hat - sigma ** 2) / np.sqrt(xi_sq_hat_2) * np.sqrt(N * T))

    return out


def run_adaptive(tau, seed=1234, print_epochs=100, fs_pct=0., all_Ys=None, num_mc=100,
                    N=100, T_max=50, t0=3, scale = 10e8, adaptive=False, adj_N=None, print_out=True, prec_thres=10, sigma=1, dp_scale_N=1):
    np.random.seed(seed)
    all_ws = [(t - 50) / 50 for t in range(0, 100)]
    if adaptive == False:
        fs_pct = 0
    if fs_pct == 0:
        adaptive = False

    if all_Ys is not None:
        num_mc = len(all_Ys)
        N, T_max = all_Ys[0].shape
    else:
        all_Ys = list()
        for j in range(num_mc):
            this_Y = np.random.normal(size=(N, 1)).dot(np.ones((1,T_max))) + np.ones((N,1)).dot(np.random.normal(size=(1, T_max))) + np.random.normal(size=(N, T_max)) * sigma
            all_Ys.append(this_Y)

    bm_treat_avg = [(2 * t + 1) / (2 * T_max) for t in range(T_max)]
    ad_treat_avg = bm_treat_avg.copy()
    half_ad_N = int(N * (1-fs_pct)/2)
    ad_N = half_ad_N * 2
    fs_N = N - ad_N
    fs_Z = calc_cv_z_mtrx(fs_N, T_max, bm_treat_avg, cv=1)
    if adj_N is None:
        adj_N = N
    scaled_Phi_list = calc_scaled_Phi_list(bm_treat_avg, T_max, adj_N)

    out = {'tau_adaptive': [], 'tau_bm': [], 'tau_oracle': [], 'tau_oracle_alt': [], 'tau_err_std': [], 'sigma_err_1_std': [],
           'sigma_err_2_std': [], 'T_ast': []}


    for j in range(num_mc):
        if (j + 1) % print_epochs == 0:
            print("{}/{} done".format(j + 1, num_mc))

        this_ctrl_Y = all_Ys[j]
        fs_Y = this_ctrl_Y[:fs_N,:] + (1+fs_Z) * tau
        ad_ctrl_Y_1 = this_ctrl_Y[fs_N:(fs_N+half_ad_N),:]
        ad_ctrl_Y_2 = this_ctrl_Y[(fs_N+half_ad_N):,:]

        for t in range(t0, T_max+1):
            ad_Z_1 = calc_cv_z_mtrx(half_ad_N, T_max, ad_treat_avg, cv=1)
            ad_Y_1 = ad_ctrl_Y_1 + (1+ad_Z_1) * tau

            ad_Z_2 = calc_cv_z_mtrx(half_ad_N, T_max, ad_treat_avg, cv=1)
            ad_Y_2 = ad_ctrl_Y_2 + (1 + ad_Z_2) * tau
            tau_hat, tau_hat_var, eps_hat = est_within(ad_Y_1[:,:t], ad_Z_1[:,:t])

            Phi = calc_Phi(fs_Z[:, :t])
            sigma_sq_hat = calc_sigma_sq(eps_hat)
            prec_ad_1 = Phi * N * t / sigma_sq_hat
            if prec_ad_1 > prec_thres:
                break

            if adaptive & (t < T_max - 1):
                tau_hat, tau_hat_var, eps_hat = est_within(fs_Y[:,:t],fs_Z[:,:t])

                sigma_sq_hat = calc_sigma_sq(eps_hat)
                xi_dagger_sq_hat = calc_sigma_sq_hat_var(eps_hat)
                scaled_xi_dagger_sq_hat = xi_dagger_sq_hat / (fs_N * t)
                Pt = get_Pt(sigma_sq_hat, scaled_xi_dagger_sq_hat, N, t, scaled_Phi_list, prec_thres)

                ad_w_t = list(np.array(ad_treat_avg[:t]) * 2 - 1)

                opt_w = dp_next_w(ad_w_t, Pt, T_max, t, N0=int(half_ad_N/dp_scale_N), scale=scale)

                ad_treat_avg[t] = (1 + opt_w)/2

        T_ast = t
        Z = np.concatenate((fs_Z, ad_Z_1, ad_Z_2), axis=0)
        Y = np.concatenate((fs_Y, ad_Y_1, ad_Y_2), axis=0)
        tau_hat, tau_hat_var_all, eps_hat_all = est_within(Y[:,:T_ast], Z[:,:T_ast])


        _, _, eps_hat_1 = est_within(ad_Y_1[:,:T_ast], ad_Z_1[:,:T_ast])
        sigma_sq_hat_1 = calc_sigma_sq(eps_hat_1)
        xi_dagger_sq_hat_1 = calc_sigma_sq_hat_var(eps_hat_1)
        sigma_err_1_std = (sigma_sq_hat_1 - sigma ** 2) / np.sqrt(xi_dagger_sq_hat_1) * np.sqrt(half_ad_N * T_ast)


        _, _, eps_hat_2 = est_within(ad_Y_2[:,:T_ast], ad_Z_2[:,:T_ast])
        sigma_sq_hat_2 = calc_sigma_sq(eps_hat_2)
        xi_dagger_sq_hat_2 = calc_sigma_sq_hat_var(eps_hat_2)
        sigma_err_2_std = (sigma_sq_hat_2 - sigma ** 2) / np.sqrt(xi_dagger_sq_hat_2) * np.sqrt(half_ad_N * T_ast)

        tau_hat_var = calc_tau_hat_var(Z[:,:T_ast], eps_hat_2)
        tau_err_std = (tau_hat - tau) / np.sqrt(tau_hat_var) * np.sqrt(N * T_ast)

        this_ctrl_Y = all_Ys[j]
        bm_treat_avg = [(2 * t + 1) / (2 * T_max) for t in range(T_max)]
        Z = np.concatenate((calc_cv_z_mtrx(fs_N, T_max, bm_treat_avg, cv=1),
                            calc_cv_z_mtrx(half_ad_N, T_max, bm_treat_avg, cv=1),
                            calc_cv_z_mtrx(half_ad_N, T_max, bm_treat_avg, cv=1)), axis=0)
        Y = this_ctrl_Y + (1 + Z) * tau
        tau_hat_bm, _, _ = est_within(Y[:, :T_ast], Z[:, :T_ast])

        opt_treat_avg = [(2 * t + 1) / (2 * T_ast) for t in range(T_ast)]
        Z_opt = np.concatenate((calc_cv_z_mtrx(fs_N, T_ast, opt_treat_avg, cv=1),
                            calc_cv_z_mtrx(half_ad_N, T_ast, opt_treat_avg, cv=1),
                            calc_cv_z_mtrx(half_ad_N, T_ast, opt_treat_avg, cv=1)), axis=0)
        Y = this_ctrl_Y[:,:T_ast] + (1 + Z_opt) * tau
        tau_hat_opt, _, _ = est_within(Y, Z_opt)

        out['tau_adaptive'].append(tau_hat - tau)
        out['tau_bm'].append(tau_hat_bm - tau)
        out['tau_oracle'].append(tau_hat_opt - tau)

        init_t = [(2 * t + 1) / (2 * T_max) for t in range(t0)]
        if t0 == T_ast:
            opt_treat_avg = init_t
        else:
            init_w = list(np.array(init_t) * 2 - 1)
            res = solve_nonadaptive_opt_design(T_ast, fix_y=init_w)
            opt_treat_avg = list((1 + res.x) / 2)
        Z_opt = calc_cv_z_mtrx(N, T_ast, opt_treat_avg, cv=1)
        Y = this_ctrl_Y[:, :T_ast] + (1 + Z_opt) * tau
        tau_hat_opt_alt, _, _ = est_within(Y, Z_opt)
        out['tau_oracle_alt'].append(tau_hat_opt_alt - tau)


        out['T_ast'].append(T_ast)
        out['tau_err_std'].append(tau_err_std)
        out['sigma_err_1_std'].append(sigma_err_1_std)
        out['sigma_err_2_std'].append(sigma_err_2_std)

        if print_out:
            print(j, "adaptive", tau_hat - tau, "bm", tau_hat_bm - tau, "oracle", tau_hat_opt - tau, "oracle_alt", tau_hat_opt_alt - tau, "T_ast", T_ast)

    return out