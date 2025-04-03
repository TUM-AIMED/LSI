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
    sorted_data = sorted(zip(kls, idxs), key=lambda x: x[0])
    kl_data, idx = zip(*sorted_data)

    return list(kl_data)



def main():
    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    paths = [
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
        "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    ]

    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]

    # paths = [
    #     "kl_jax_torch_1000_remove_4646_dataset_Primacompressed_subset_4646_range_0_4646_corrupt_0.0_torch.pkl"
    # ]
    paths = [base_path + path for path in paths]  
    # paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_1000_dataset_Imagewoofcompressed_subset_9025_range_0_9025_corrupt_0.0_torch.pkl"] 
    # paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_1000_dataset_Imagenettecompressed_subset_9469_range_0_9469_corrupt_0.0_torch.pkl"]  
    # paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_1000_dataset_cifar10compressed_model_Tinymodel_subset_10000_range_0_9999_corrupt_0.0_corrupt_data_0.0_0_torch.pkl"]
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"

    kl_data = []
    idx_data = []
    noise_data = []
    clip_data = []
    # clip_data = [0.01, 0.1, 1.0, 5.0] 
    # clip_data = [0.1, 0.1, 0.1, 0.1]

    # filenames.reverse()
    # clip_data.reverse()

    kl_data = get_kl_data(paths)
    


    # Create a 2x2 subplot grid
    # fig = plt.figure(figsize=(2, 1.7))
    fig = plt.figure(figsize=(3.5, 2))
    # Flatten the subplot array for easy indexing
    # axs = axs.flatten()

    # Iterate through rows of the data and create histograms in subplots
    long_df = pd.DataFrame()

    temp_df = pd.DataFrame({'Value': kl_data})
    p = sns.kdeplot(data=temp_df, log_scale=False, fill=True, common_norm=False, linewidths=2, palette='viridis', legend=False, clip=(0, 0.45))

    # for i in range(len(kl_data)):
    #     sns.kdeplot(data=kl_data[i], log_scale=True, ax=axs, fill=True, alpha=0.5)
        # axs[i].hist(kl_data[i], bins=logbins, color='slateblue', edgecolor='black', linewidth=1)
        

    # Adjust layout for better spacing
    # plt.xlim(1e-5, 5e4)
    # plt.ylim(0, 3.2)
    # plt.xlim(0e-5, 8e-1)
    # plt.ylim(0, 1.5)
    plt.xlabel("LSI")
    plt.ylabel("Density")
    # plt.ylabel(None)
    plt.title("CIFAR10")
    # plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)
    # sns.move_legend(p, "upper right", frameon=False)
    plt.tight_layout()
    save_name = save_dir + "hists_lis_sns_Cifar10_not_subset_compressed"
    plt.savefig(save_name + ".png", format="png", dpi=1000)
    print(f"saving fig as {save_name}.png")
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")

    print("")
    
main()