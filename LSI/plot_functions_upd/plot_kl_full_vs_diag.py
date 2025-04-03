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

def get_kl_data(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])
    kl_data = kl_data.mean(axis=0)
    return kl_data


def main():
    path_str1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_one_day_bf_deadline/kl_jax_torch_1000_remove_500_dataset_Primacompressed_subset_4646_range_0_500_corrupt_0.0_used_for_hessian_comp_this_is_diag.pkl"
    path_str2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_one_day_bf_deadline/kl_jax_torch_1000_remove_500_dataset_Primacompressed_subset_4646_range_0_500_corrupt_0.0_used_for_hessian_comp_this_is_full.pkl"

    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"
   
   
    kl1_diag = get_kl_data(path_str1)
    kl1_full = get_kl_data(path_str2)

    kl1_diag = kl1_diag[0:len(kl1_full)]
    third = [1 for i in range(len(kl1_diag))]
    fig = plt.figure(figsize=(3.5, 3.5))

    df = pd.DataFrame({'diag': kl1_diag, 'full': kl1_full, 'lab':third})

    sns.scatterplot(df, x="diag", y="full", palette='viridis', hue="lab", legend=False)

    plt.xlabel("LSI computed on Diagonal Hessian")
    plt.ylabel("LSI computed on Full Hessian")

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)
    plt.tight_layout()
    save_name = save_dir + "diag_vs_full"
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")

    print("")
    
main()