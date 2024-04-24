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
    square_sum = np.array(final_dict["square_diff"])
    return kl_data, square_sum


def main():
    # path_str1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_10_remove_100_dataset_cifar10compressed_subset_50000_corrupt_0.0_with_square_diff.pkl"
    path_str1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_100_dataset_cifar10compressed_subset_50000_corrupt_0.0_with_square_diff.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
   
   
    kl, square_sum = get_kl_data(path_str1)
    

    fig = plt.figure(figsize=(3.5, 3.5))

    df = pd.DataFrame({'Kl': kl, 'Square distance weights': square_sum})

    sns.scatterplot(df, x='Kl', y='Square distance weights', palette='viridis', legend=False)

    plt.xlabel("LSI")
    plt.ylabel("squared-difference of the weights")

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)
    plt.tight_layout()
    save_name = save_dir + "kl_vs_weight_diff"
    plt.savefig(save_name + ".png", format="png", dpi=1000)
    print(f"saving fig as {save_name}.png")

    print("")
    
main()