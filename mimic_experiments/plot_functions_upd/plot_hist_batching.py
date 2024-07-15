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



def get_kl_data(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])[0]
    batches = final_dict["idx_batch_assignmet"] 
    batches = [idx.tolist() for idx in batches]
    kl_data = [kl_data[batch] for batch in batches]
    return kl_data



def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/kl_jax_torch_400_remove_4646_dataset_Primacompressed_subset_4646_range_0_4646_corrupt_0.0_batched_3_torch.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"

    batches = 5

    kl1_diag = get_kl_data(kl_path)
    

    # Create a 2x2 subplot grid
    fig = plt.figure(figsize=(3.5, 3.5))
    # Flatten the subplot array for easy indexing
    # axs = axs.flatten()

    # Iterate through rows of the data and create histograms in subplots
    long_df = pd.DataFrame()

    for i, data_array in enumerate(kl1_diag):
        temp_df = pd.DataFrame({'Value': data_array, 'Label': "Batch " + str(i)})
        long_df = pd.concat([long_df, temp_df])
    p = sns.kdeplot(data=long_df, log_scale=True, fill=True, alpha=0.5, hue="Label", x="Value", common_norm=False, linewidths=2, palette='viridis')

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

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)
    sns.move_legend(p, "upper left", frameon=False, title=None)
    plt.tight_layout()
    save_name = save_dir + "hists_batch"
    plt.savefig(save_name + ".png", format="png", dpi=1000)
    print(f"saving fig as {save_name}.png")

    print("")
    
main()