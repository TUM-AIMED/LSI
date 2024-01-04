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
from tqdm import tqdm


plt.rcParams.update({
    'font.family': 'serif',
})

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    idx = final_dict["idx"][0]
    kl_data_list = final_dict["kl"][0]
    return kl_data_list, idx, None



def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_.pkl"
    file_name = "z_boxplot_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000"
    rand = True
    kl1_diag, init_idx, rand_labels_idx = get_kl_data(kl_path, "kl1_diag", rand=rand)

    kl1_diag = kl1_diag

    combined_data = list(zip(kl1_diag, init_idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl1_diag, idx = zip(*sorted_data)

    plt.figure(figsize=(50, 6))
    plt.scatter([*range(len(kl1_diag))], kl1_diag)

    plt.xlabel("Index")
    plt.ylabel("KL1 - Diag")

    plt.tight_layout()
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

main()