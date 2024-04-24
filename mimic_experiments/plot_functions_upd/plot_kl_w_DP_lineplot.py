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

def one_hot_encoding(labels):
    lb = LabelBinarizer()
    one_hot_encoded = lb.fit_transform(labels)
    return lb.classes_, one_hot_encoded.tolist()


plt.rcParams.update({
    'font.family': 'serif',
})

def get_n_distinct_colors(n):
    # Get a list of distinct colors from the TABLEAU_COLORS colormap
    if n > 10:
        return ['#e377c2' for i in range(n)]
    all_colors = list(TABLEAU_COLORS.values())
    distinct_colors = list(TABLEAU_COLORS.values())[:n]
    return distinct_colors

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)



def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])
    # kl_data = np.mean(kl_data, axis=0)
    kl_data = kl_data.flatten()
    idx = final_dict["idx"][0]
    noise = final_dict["noise"]
    clip = final_dict["clip"]
    return kl_data, idx, noise, clip



def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_dp_upd2_fixed_noise_schedule/"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
    # filenames = [
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_5.0_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_1.0_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.1_.pkl",
    #     "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.01_.pkl"
    #     ]
    filenames = [
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_0.0_clip_0.1_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_8.1_clip_0.1_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_16.9_clip_0.1_.pkl",
        "kl_jax_epochs_700_lr_2_remove_5000_seeds_5_dataset_cifar10compressed_subset_50000_noise_25.5_clip_0.1_.pkl"
    ]
    
    kl_data = []
    idx_data = []
    noise_data = []
    clip_data = []
    # clip_data = [0.01, 0.1, 1.0, 5.0] 
    # clip_data = [0.1, 0.1, 0.1, 0.1]

    # filenames.reverse()
    # clip_data.reverse()

    for filename in filenames:
        kl_path = path_str + filename
        kl1_diag, init_idx, noise, clip = get_kl_data(kl_path, "kl1_diag")
        kl_data.append(kl1_diag)
        idx_data.append(init_idx)
        noise_data.append(noise)
        clip_data.append(clip)
    
    for inidices in idx_data:
        assert inidices == idx_data[0]

    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the subplot array for easy indexing
    axs = axs.flatten()

    # Iterate through rows of the data and create histograms in subplots
    for i in range(len(kl_data)):
        hist, bins = np.histogram(np.array(kl_data).flatten(), bins=8)
        print(bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),150)
        axs[i].hist(kl_data[i], bins=logbins, color='slateblue', edgecolor='black', linewidth=1)
        axs[i].set_title(f'Noise {noise_data[i]} Clip {clip_data[i]}')  # Set title for each subplot
        if i == 0 or i == 2:
            axs[i].set_ylabel("Count")
        if i == 2 or i == 3:
            axs[i].set_xlabel("KL-divergence Value")

    # Adjust layout for better spacing
    for ax in axs:
        # ax.set_xlim(left=0, right=1)
        # ax.set_ylim(bottom=0, top=1e4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([4e-5, 1e0])  # Set your desired x-limits
        ax.set_ylim([1, 5e3])  # Set your desired y-limits
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.suptitle('Impact of differnt norm clipping thresholds on the sample informativeness', fontsize=16)

    save_name = save_dir + "hists_dp"
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")

    print("")
    
main()