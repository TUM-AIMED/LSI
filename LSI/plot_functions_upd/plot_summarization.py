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
import ast
from matplotlib.colors import TABLEAU_COLORS
from sklearn.preprocessing import LabelBinarizer
import random
from tqdm import tqdm
import seaborn as sns
from matplotlib.cm import viridis
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



def get_kl_data(final_path, agg_type="", rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    selector = "test_full_acc"
    low = final_dict[0][selector]
    high = final_dict[1][selector]
    rand = final_dict[2][selector]
    low = [np.mean(list(subdict.values())) for subdict in low.values()]
    high = [np.mean(list(subdict.values())) for subdict in high.values()]
    rand = [np.mean(list(subdict.values())) for subdict in rand.values()]
    keys = final_dict[0][selector].keys()

    return [low, high, rand], keys


def main():
    filename = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_summarization/kl_jax_epochs_1000_dataset_cifar10compressed_subset_50000_summarization.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"


    portions = []
    train_subset_cwa, portions= get_kl_data(filename)
    portions = np.array(list(portions)).astype(float)
    portions = (5000 - portions) / 5000
    # train_subset_cwas[combination] = train_subset_cwa
    # data_new = train_subset_cwa


    data = pd.DataFrame(np.array(train_subset_cwa).transpose(), columns=["Remove low LSI", "Remove high LSI", "Remove random"])
    labels = pd.DataFrame(portions, columns=["portions"])
    data = pd.concat([data, labels], axis=1)
    label_list = ["Remove low LSI", "Remove high LSI", "Remove random"]
    long_df = pd.melt(data, id_vars="portions", var_name='type', value_name='value')

    
    portions = np.char.add(np.char.mod('%.1f', portions*100), '%')

    fig = plt.figure(figsize=(3.5, 2.5))
    # [none, random, first, last]
    p = sns.lineplot(data=long_df, y="value", x="portions", palette='viridis', hue="type")

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)

    # plt.xticks(np.arange(len(portions))[::2], portions[::2], rotation='vertical')
    # plt.ylim([0, 0.8])
    plt.xlabel("Portion of the Intial Sample Count")
    plt.ylabel("Test Accuracy")
    sns.move_legend(p, "lower left", frameon=False, title=None)
    plt.tight_layout()
    save_name = save_dir + "summ_subset"
    plt.savefig(save_name + ".pdf", format="pdf")
    print(f"saving fig as {save_name}.pdf")



    
main()