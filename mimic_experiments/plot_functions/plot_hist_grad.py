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


plt.rcParams.update({
    'font.family': 'serif',
})

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_grad_data(final_path):
    with open(final_path, 'rb') as file:
        grad_data = pickle.load(file)
    idxs = []
    result = []
    for seed, seed_data in grad_data.items():
        seed_wise_results = []
        idx = []
        for index, idx_data in seed_data.items():
            idx.append(index)
            seed_wise_results.append(np.sqrt(np.sum(idx_data)))
        result.append(seed_wise_results)
        idxs.append(idx)
    if not all(inner_list == idxs[0] for inner_list in idxs):
        raise Exception("some mix up")
    return np.array(idxs[0]), np.mean(result, axis=0)



def main():
    grad_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_cifar100compressed_logreg4_1_100.0_1e-07__1000_50000/results.pkl"
    file_name = "z_hist_grad_cifar100compressed_logreg4_1_100.0_1e-07__1000_50000"
    idx, grad_data_mean = get_grad_data(grad_path)

    plt.figure(figsize=(8, 6))
    plt.hist(grad_data_mean, bins="auto", edgecolor='blue', alpha=0.7)
    plt.xlabel("KL1 - Diag")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

main()