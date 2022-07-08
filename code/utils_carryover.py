import numpy as np


def const_obj(y):
    """
    Objective function value for the model with only direct treatment effect

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t

    Returns
    -------
    Objective function value
    """
    #
    T = len(y)
    
    y_sum = 0
    for k in range(T):
        y_sum = y_sum + y[k]
    y_weighted_sum = 0
    for k in range(T):
        y_weighted_sum = y_weighted_sum + (k+1)*y[k]
    y_sq_sum = 0
    for k in range(T):
        y_sq_sum = y_sq_sum + y[k]**2
    obj = 2*(T+1)/T*y_sum - 4/T*y_weighted_sum + y_sq_sum - 1/T*y_sum**2
    
    return obj



def theta1_offdiag(y, L, j, k):
    """
    The objective function involves two matrices, theta1 and theta2
    This function computes the off-diagonal terms in theta1

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t
    L : int
        Number of lagged periods that the treatment can affect
    j : int
        index of the row
    k : int
        index of the column

    Returns
    -------
    (j,k)-th entry in theta1
    """

    T = len(y)
    val1 = 0
    for l in range(j,k):
        val1 += y[l]-y[T-L+l]
    val = (T-L)+val1
    return val

def theta1(y, L=1):
    """
    The objective function involves two matrices, theta1 and theta2
    This function computes theta1, where theta1 is defined in the proof of Lemma 5 and 8 and is equal to \mathcal{Z}^T \mathcal{Z}

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t
    L : int
        Number of lagged periods that the treatment can affect

    Returns
    -------
    theta1
    """

    T = len(y)
    H = np.zeros((L+1, L+1))
    for j in range(L+1):
        H[j,j] = T-L
    for j in range(L):
        for k in range(j+1,L+1):
            H[j,k] = theta1_offdiag(y, L, j, k)
            H[k,j] = H[j,k]
    return H



def calc_v(y, L, j, k):
    """
    Compute the vector v that is used to compute the coefficient of the linear term in Lemma 5 and 8 and is defined in the proof of Lemma 5 and 8

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t
    L : int
        Number of lagged periods that the treatment can affect
    j : int
        First index
    k : int
        Second index

    Returns
    -------
    v
    """

    T = len(y)
    v_list = [0 for _ in range(T+1)]
    j_prime = j+1; k_prime = k+1
    for t in range(L+1-k_prime):
        v_list[t] = 1
    for t in range(L+1-k_prime, L+1-j_prime):
        v_list[t] = -(-1 + 2*(t-1-L+k_prime)/(T-L))
    for t in range(L+1-j_prime, T+1-k_prime):
        v_list[t] = (-1 + 2*(t-1-L+j_prime)/(T-L))*(-1 + 2*(t-1-L+k_prime)/(T-L))
    for t in range(T+1-k_prime, T+1-j_prime):
        v_list[t] = (-1 + 2*(t-1-L+j_prime)/(T-L))
    for t in range(T+1-j_prime, T+1):
        v_list[t] = 1
    return v_list



def theta2_offdiag(y, L, j, k):
    """
    The objective function involves two matrices, theta1 and theta2
    This function computes the off-diagonal terms in theta2

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t
    L : int
        Number of lagged periods that the treatment can affect
    j : int
        index of the row
    k : int
        index of the column

    Returns
    -------
    (j,k)-th entry in theta2
    """

    T = len(y)
    r_list = [0 for _ in range(T+1)]
    r_list[0] = (1+y[0])/2; r_list[T] = (1-y[T-1])/2
    for t in range(1,T):
        r_list[t] = (y[t]-y[t-1])/2
    v_list = calc_v(y, L, j, k)
    val2 = 0
    for t in range(T+1):
        val2 += v_list[t]*r_list[T-t]
    val2 *= (T-L)
    val3 = 0
    for t in range(j,T-L+j):
        val3 += y[t]*y[t+k-j]
    val4 = np.sum(y[j:(T-L+j)])*np.sum(y[k:(T-L+k)])/(T-L)
    val = (val2+val3-val4)
    return val



def theta2(y, L=1):
    """
    The objective function involves two matrices, theta1 and theta2
    This function computes theta2, where theta2 is defined in the proof of Lemma 5 and 8 and is equal to \mathcal{Z}^T \Gamma (\Gamma^T \Gamma)^{-1} \Gamma^T \mathcal{Z}

    Parameters
    ----------
    y : list of T floats that are between -1 and 1
        Cross section average of the design matrix, (y[t]+1)/2 is the treated proportion at time t
    L : int
        Number of lagged periods that the treatment can affect

    Returns
    -------
    theta2
    """

    T = len(y)
    H = np.zeros((L+1, L+1))
    for j in range(L+1):
        H[j,j] = (T-L)+const_obj(y[j:(T-L+j)])
    for j in range(L):
        for k in range(j+1,L+1):
            H[j,k] = theta2_offdiag(y, L, j, k)
            H[k,j] = H[j,k]
    return H
