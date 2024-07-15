import io
import pickle
import torch
import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from Datasets.dataset_helper import get_dataset
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


def get_kl_data_cf10(final_path, len_data):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])[0]
    idx = np.array(final_dict["idx"])[0]
    corrupt_idx = np.array(final_dict["corrupted_idx"]) - 10000
    idx = list(range(10000)) 
    corrupt_idx = [i for i in corrupt_idx if i in idx]
    non_corrupt_idx = [i for i in idx if i not in corrupt_idx]
    # non_corrupt_idx = [index for index in idx if idx not in corrupt_idx]
    corrupt_kl = kl_data[corrupt_idx]
    non_corrupt_kl = kl_data[non_corrupt_idx]
    return corrupt_kl, non_corrupt_kl


def get_kl_data(final_path, len_data):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])[0]
    idx = np.array(final_dict["idx"])[0]
    corrupt_idx = np.array(final_dict["corrupted_idx"])
    idx = list(range(len_data)) 
    corrupt_idx = [i for i in corrupt_idx if i in idx]
    non_corrupt_idx = [i for i in idx if i not in corrupt_idx]
    # non_corrupt_idx = [index for index in idx if idx not in corrupt_idx]
    corrupt_kl = kl_data[corrupt_idx]
    non_corrupt_kl = kl_data[non_corrupt_idx]
    return corrupt_kl, non_corrupt_kl

def main():
    path_str1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/kl_jax_torch_1000_remove_4646_dataset_Primacompressed_subset_4646_range_0_4646_corrupt_0.1_torch_data_fixed.pkl"
    path_str2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_torch_1000_remove_1000_dataset_cifar10compressed_subset_50000_range_0_1000_corrupt_0.1_torch_higher_lr_lap_on_full_data_fixed.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"

    # Extract KL data for both paths
    corrupt_kl1, non_corrupt_kl1 = get_kl_data(path_str1, len_data=4646)
    corrupt_kl2, non_corrupt_kl2 = get_kl_data(path_str2, len_data=1000)

    # Prepare data for the first plot
    kl_data1 = [corrupt_kl1, non_corrupt_kl1]
    label_list = ["Corrupted Labels", "Uncorrupted Labels"]
    long_df1 = pd.DataFrame()

    for i, data_array in enumerate(kl_data1):
        temp_df = pd.DataFrame({'Value': data_array, 'Label': label_list[i]})
        long_df1 = pd.concat([long_df1, temp_df])

    # Prepare data for the second plot
    kl_data2 = [corrupt_kl2, non_corrupt_kl2]
    long_df2 = pd.DataFrame()

    for i, data_array in enumerate(kl_data2):
        temp_df = pd.DataFrame({'Value': data_array, 'Label': label_list[i]})
        long_df2 = pd.concat([long_df2, temp_df])

    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 2))

    # Set common plot parameters
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.rc('axes', titlesize=8)

    # Plot the first KDE plot
    p1 = sns.kdeplot(data=long_df1, log_scale=True, fill=True, alpha=0.5, hue="Label", x="Value", common_norm=False, linewidths=2, palette='viridis', ax=axs[0], legend=False)
    axs[0].set_title('Pneumonia', fontsize=8)
    axs[0].set_xlabel("LSI", fontsize=8)
    axs[0].set_ylabel("Density", fontsize=8)

    # Plot the second KDE plot
    p2 = sns.kdeplot(data=long_df2, log_scale=True, fill=True, alpha=0.5, hue="Label", x="Value", common_norm=False, linewidths=2, palette='viridis', ax=axs[1], legend=True)
    axs[1].set_title('CIFAR-10', fontsize=8)
    axs[1].set_xlabel("LSI", fontsize=8)
    axs[1].set_ylabel(None, fontsize=8)
    sns.move_legend(p2, "upper left", bbox_to_anchor=(1, 1))
    p2.legend_.set_title(None)

    plt.tight_layout()
    save_name = save_dir + "new_corruption_both"
    plt.savefig(save_name + ".png", format="png", dpi=1000)
    print(f"saving fig as {save_name}.png")
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")



    
main()