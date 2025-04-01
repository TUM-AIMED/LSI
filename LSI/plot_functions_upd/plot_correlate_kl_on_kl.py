import os
import io
import pickle
import torch
import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib import animation
from Datasets.dataset_helper import get_dataset
import cv2
from collections import defaultdict
import ast
from matplotlib.colors import TABLEAU_COLORS
from sklearn.preprocessing import LabelBinarizer
import random
from tqdm import tqdm


plt.rcParams.update({
    'font.family': 'serif',
})


def get_kl_data(paths):
    idxs = []
    kls = []
    for batch, path in enumerate(paths):
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
            kl_data = np.array(final_dict["kl"])
            kl_data = np.mean(kl_data, axis=0)
            idx = list(range(len(kl_data)))
            idx = [idx_i + 10000 * batch for idx_i in idx]
            kls.extend(kl_data)
            idxs.extend(idx)
    # sorted_data = sorted(zip(kls, idxs), key=lambda x: x[0])
    # kl_data, idx = zip(*sorted_data)

    return list(kls), idxs

def main():
    # base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]
    # paths1 = [base_path + path for path in paths] 

    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/"
    paths = [
        "kl_jax_torch_700_remove_1000_dataset_cifar10_model_CNN_subset_10000_range_0_1000_corrupt_0.0__training_on_full_network.pkl"
    ]
    paths1 = [base_path + path for path in paths]

    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/"
    paths = [
        "kl_jax_torch_700_remove_1000_dataset_cifar10_model_MLP_subset_10000_range_0_1000_corrupt_0.0__training_on_full_network.pkl"
    ]
    paths2 = [base_path + path for path in paths]  


    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
    kl1, idx1 = get_kl_data(paths1)
    kl2, idx2 = get_kl_data(paths2)
    kl1 = kl1[0 :len(kl2)]

    plt.scatter(kl1, kl2)

    plt.tight_layout()
    save_name = save_dir + "corr_single_CNN"
    plt.savefig(save_name + ".png", format="png")
    print(f"saving fig as {save_name}.png")



    
main()