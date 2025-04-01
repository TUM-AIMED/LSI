
import os
import pickle
from tqdm import tqdm
from utils.data_logger import computeKL
from copy import deepcopy
import numpy as np

from collections import defaultdict 

folders = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_replace_cifar_logged_grads2500_idx_0"]

final_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_replace_cifar_logged_grads2500_idx_0_kl"

all_files = []
for folder in folders:
    files =  os.listdir(folder)
    files = [folder + "/" + file for file in files]
    all_files.extend(files)


# kl1_all = {}
# kl2_all = {}
# kl1_all_max = {}
# kl2_all_max = {}
# for file_invest in tqdm(all_files):
#     kl_1_indiv = []
#     kl_2_indiv = []
#     with open(file_invest, 'rb') as file_data_invest:
#         data_invest = pickle.load(file_data_invest)
#         data_invest = data_invest[-1]
#         data_invest_mean = data_invest["laplace_approx_mean"]
#         data_invest_prec = data_invest["laplace_approx_precision"]
#         removed_idx = data_invest["removed_idx"]

#     for file_compare in tqdm(all_files):
#         if file_compare == file_invest:
#             continue
#         with open(file_compare, 'rb') as file_data_compare:
#             data_compare = pickle.load(file_data_compare)
#             data_compare = data_compare[-1]
#             data_compare_mean = data_compare["laplace_approx_mean"]
#             data_compare_prec = data_compare["laplace_approx_precision"]
#         kl1 = computeKL(data_invest_mean, data_compare_mean, data_invest_prec, data_compare_prec)
#         kl2 = computeKL(data_compare_mean, data_invest_mean, data_compare_prec, data_invest_prec)
#         kl_1_indiv.append(kl1)
#         kl_2_indiv.append(kl1)
#     kl1_all[removed_idx] = np.mean(kl_1_indiv)
#     kl2_all[removed_idx] = np.mean(kl_2_indiv)
#     kl1_all_max[removed_idx] = np.max(kl_1_indiv)
#     kl2_all_max[removed_idx] = np.max(kl_2_indiv)
# results = {}
# results["kl1_mean"] = kl1_all
# results["kl2_mean"] = kl2_all
# results["kl1_max"] = kl1_all_max
# results["kl2_max"] = kl2_all_max
# if not os.path.exists(final_path):
#     os.makedirs(final_path)
# with open(final_path + "/" + "final.pkl", 'wb') as file:
#     pickle.dump(results, file)
# print(kl1_all)

all_files_compare = deepcopy(all_files)
dict = defaultdict()
for file_invest in tqdm(all_files):
    kl_1_indiv = []
    kl_2_indiv = []
    with open(file_invest, 'rb') as file_data_invest:
        data_invest = pickle.load(file_data_invest)
        data_invest = data_invest[-1]
        data_invest_mean = data_invest["laplace_approx_mean"]
        data_invest_prec = data_invest["laplace_approx_precision"]
        removed_idx = data_invest["removed_idx"]
        for file_compare in tqdm(all_files_compare):
            if file_compare == file_invest:
                continue
            with open(file_compare, 'rb') as file_data_compare:
                data_compare = pickle.load(file_data_compare)
                data_compare = data_compare[-1]
                data_compare_mean = data_compare["laplace_approx_mean"]
                data_compare_prec = data_compare["laplace_approx_precision"]
                compare_idx = data_compare["removed_idx"]
            kl1 = computeKL(data_invest_mean, data_compare_mean, data_invest_prec, data_compare_prec)
            kl2 = computeKL(data_compare_mean, data_invest_mean, data_compare_prec, data_invest_prec)
            if str(removed_idx) not in dict:
                dict[str(removed_idx)] = {}
            if "kl1" not in dict[str(removed_idx)]:
                dict[str(removed_idx)]["kl1"] = {}
            if "kl2" not in dict[str(removed_idx)]:
                dict[str(removed_idx)]["kl2"] = {}
            if str(compare_idx) not in dict:
                dict[str(compare_idx)] = {}
            if "kl1" not in dict[str(compare_idx)]:
                dict[str(compare_idx)]["kl1"] = {}
            if "kl2" not in dict[str(compare_idx)]:
                dict[str(compare_idx)]["kl2"] = {}
            dict[str(removed_idx)]["kl1"][compare_idx] = kl1
            dict[str(removed_idx)]["kl2"][compare_idx] = kl2
            dict[str(compare_idx)]["kl1"][removed_idx] = kl2
            dict[str(compare_idx)]["kl2"][removed_idx] = kl1
    all_files_compare.remove(file_invest)

kl1_all_max = {}
kl2_all_max = {}
kl1_all_mean = {}
kl2_all_mean = {}

for idx, kl1kl2 in dict.items():
    kl1 = list(kl1kl2["kl1"].values())
    kl2 = list(kl1kl2["kl2"].values())
    kl1_all_max[idx] = np.max(np.array(kl1))
    kl2_all_max[idx] = np.max(np.array(kl2))
    kl1_all_mean[idx] = np.mean(np.array(kl1))
    kl2_all_mean[idx] = np.mean(np.array(kl2))
results = {}
results["kl1_mean"] = kl1_all_mean
results["kl2_mean"] = kl2_all_mean
results["kl1_max"] = kl1_all_max
results["kl2_max"] = kl2_all_max
if not os.path.exists(final_path):
    os.makedirs(final_path)
with open(final_path + "/" + "final.pkl", 'wb') as file:
    pickle.dump(results, file)


print("end")