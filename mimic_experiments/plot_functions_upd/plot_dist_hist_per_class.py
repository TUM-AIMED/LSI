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



def get_kl_data(paths, labels):
    idxs = []
    kls = []
    kls_by_class = []
    labels_by_class = []
    for batch, path in enumerate(paths):
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
            kl_data = np.array(final_dict["kl"])
            kl_data = np.mean(kl_data, axis=0)
            idx = list(range(len(kl_data)))
            idx = [idx_i + 10000 * batch for idx_i in idx]
            kls.extend(kl_data)
            idxs.extend(idx)
    for label_num in labels.unique():
        label_list_kl = []
        label_list_idx = []
        for index, kl, lab in zip(idxs, kls, labels):
            if lab == label_num:
                label_list_kl.append(abs(kl))
                label_list_idx.append(index)
        kls_by_class.append(label_list_kl)
        labels_by_class.append(label_num.item())
    return list(kls_by_class), labels_by_class



def main():
    base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
    # paths = [
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    #     "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
    # ]

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
    # paths = [base_path + path for path in paths]  
    paths = [["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_4646_dataset_Primacompressed_model_Tinymodel_subset_4646_range_0_4646_corrupt_0.0_corrupt_data_0.0_1_torch.pkl"],
             ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_4646_dataset_Primacompressed_model_Tinymodel_subset_4646_range_0_4646_corrupt_0.0_corrupt_data_0.5_1_torch.pkl"],
             ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_4646_dataset_Primacompressed_model_Tinymodel_subset_4646_range_0_4646_corrupt_0.0_corrupt_data_1.0_1_torch.pkl"],
             ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_4646_dataset_Primacompressed_model_Tinymodel_subset_4646_range_0_4646_corrupt_0.0_corrupt_data_5.0_1_torch.pkl"]]  
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"


    label_mapping_Prima = {
        0: "Bacterial",
        1: "Normal",
        2: "Viral"
    }

    data_noise = [0.0,
                  0.5,
                  1.0,
                  5.0]

    dataset_class, data_path = get_dataset("Primacompressed") # Prima
    data_set = dataset_class(data_path, train=True)
    labels = data_set.labels

    kl_data_all = []

    for path_set in paths:
        kl_data, labels_order = get_kl_data(path_set, labels)
        transposed_data = list(map(list, zip(*kl_data)))

        labels_str = [label_mapping_Prima[lab] for lab in labels_order]
        kl_data_all.append(transposed_data)

    fig, axes = plt.subplots(1, 4, figsize=(7, 2))
    # fig = plt.figure(figsize=(3.5, 2))

    for i, data in enumerate(kl_data_all):
        draw_legend = True if i == 3 else False
        long_df = pd.DataFrame(data, columns=labels_str).astype("float64")
        p = sns.kdeplot(data=long_df, log_scale=True, gridsize=100000, fill=True, common_norm=False, linewidths=2, palette='viridis', legend=draw_legend, ax=axes[i])
        if draw_legend:
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1))
        axes[i].set_title(f'Data Noise {data_noise[i]}', fontsize=8)
        axes[i].set_xlabel("LSI", fontsize=8)
        axes[i].set_ylabel("Density", fontsize=8) if i == 0 else axes[i].set_ylabel("")
        if i != 0: axes[i].set_yticklabels([])
        axes[i].set_xlim((1e-6, 1e3))
        axes[i].set_xticks((1e-4, 1e-2, 1e0, 1e2))
        axes[i].set_ylim((0, 1.1))
        axes[i].grid(True)
        for label in axes[i].get_xticklabels():
            label.set_fontsize(7.5)

 
    # sns.move_legend(p, "upper right", frameon=False)
    plt.subplots_adjust(wspace=-1)
    plt.tight_layout()
    save_name = save_dir + "hists_lis_sns_Prima_per_class_corr000"
    plt.savefig(save_name + ".png", format="png", dpi=100)
    print(f"saving fig as {save_name}.png")
    plt.savefig(save_name + ".pdf", format="pdf", dpi=100)
    print(f"saving fig as {save_name}.pdf")

    print("")
    
main()