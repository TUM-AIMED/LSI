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
from matplotlib.colors import TABLEAU_COLORS
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import pandas as pd
# Set Seaborn theme with paper context and font scale 2 (or 1.5)
sns.set_theme(context="paper", font_scale=2)

# Remove spines on every figure
sns.despine()

# Set colormap to "viridis" or another colorblind-friendly one
cmap = "viridis"

# Set minimum linewidth to 2
sns.set_context("paper", rc={"lines.linewidth": 2})

plt.rcParams.update({
    'font.family': 'serif',
})

def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])
    # kl_data = np.mean(kl_data, axis=0)
    kl_data = kl_data.mean(axis=0)
    idx = final_dict["idx"][0]
    noise = final_dict["noise"]
    clip = final_dict["clip"]
    return kl_data, idx, noise, clip



def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_dp_upd2_fixed_noise_schedule/"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
    filenames = [
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_5.0_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_1.0_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.1_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.01_.pkl"
        ]
    # filenames = [
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.1_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_8.1_clip_0.1_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_16.9_clip_0.1_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_25.5_clip_0.1_.pkl"
    # ]
    
    kl_data = []
    idx_data = []
    noise_data = []
    clip_data = []
    # clip_data = [0.01, 0.1, 1.0, 5.0] 
    # clip_data = [0.1, 0.1, 0.1, 0.1]

    # filenames.reverse()
    # clip_data.reverse()

    first_array = kl_data[0]
    indices = np.argsort(first_array)[:500]

    kl_data_2 = []
    for data in kl_data:
        kl_data_2.append(data[indices])
    kl_data = kl_data_2

    for filename in filenames:
        kl_path = path_str + filename
        kl1_diag, init_idx, noise, clip = get_kl_data(kl_path, "kl1_diag")
        kl_data.append(kl1_diag)
        idx_data.append(init_idx)
        noise_data.append(noise)
        clip_data.append(clip)
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))

    # Iterate through pairs of arrays and create scatter plots
    for i in range(3):
        axs[i].scatter(kl_data[i], kl_data[i + 1])
        axs[i].set_title(f'Array {i+1}')


    save_name = save_dir + "ordering"
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")

    print("")
    
main()