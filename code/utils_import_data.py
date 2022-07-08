import numpy as np

def import_MS_FLU_pos_data():
    Y = np.loadtxt("../data/flu.csv", delimiter=',')
    return Y

def import_medical_data():
    # import data
    Y = np.loadtxt("../data/medical.csv", delimiter=',')
    return Y

def import_supermarket_data():
    # import data
    Y = np.loadtxt("../data/grocery.csv", delimiter=',')
    return Y

def import_loan_data():
    Y = np.loadtxt("../Data/loan.csv", delimiter=',')
    return Y