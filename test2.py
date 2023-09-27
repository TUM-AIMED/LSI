import pickle
import matplotlib.pyplot as plt
import numpy as np

def compute_log_det(diag_array):
    det = np.sum(np.log(diag_array))
    return det



def computeKL(mean1, mean2, precision1, precision2):
    inv_precision1 = 1/precision1
    inv_precision2 = 1/precision2
    mean_difference = mean2 - mean1
    test1 = np.sum(np.multiply(precision2, inv_precision1)) 
    test2 = np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
    test3 = len(mean1) 
    test4 = compute_log_det(inv_precision2) - compute_log_det(inv_precision1)
    test5 = compute_log_det(inv_precision2)
    test6 = compute_log_det(inv_precision1)
    print("------")
    print(test1)
    print(test2)
    print(test3)
    print(test4)
    print(test5)
    print(test6)
    print("------")
    kl = 0.5*(np.sum(np.multiply(precision2, inv_precision1)) 
              + np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
              - len(mean1) 
              + compute_log_det(inv_precision2) - compute_log_det(inv_precision1))
    return kl   



    
compare_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist5/gradients/laplace_0_remove_0.pkl"
model_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist5/gradients/laplace_0_remove_1.pkl"
with open(compare_path, 'rb') as f:
    data = pickle.load(f) 
    data = data[-1]
    mean2 = data["laplace_approx_mean"]
    precision2 = data["laplace_approx_precision"]

with open(model_path, 'rb') as f:
    data = pickle.load(f) 
    data = data[-1]
    mean1 = data["laplace_approx_mean"]
    precision1 = data["laplace_approx_precision"]

# KL(model||model_compare)
test_precision = np.ones(precision1.shape)
print("---------")
kl1 = computeKL(mean1, mean2, precision1, precision2)
print(f"kl1: {kl1}")
# KL(model_compare||model)
kl2 = computeKL(mean2, mean1, precision2, precision1)
print(f"kl2: {kl2}")
print("---------")
kl_test = computeKL(mean1, mean2, test_precision, test_precision)
print(f"kl_test: {kl_test}")
print("---------")
