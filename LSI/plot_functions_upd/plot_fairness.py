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
    combination = final_dict["combination"]
    idxs = final_dict["idx_train_subset"]
    train_pred = final_dict["train_full_acc"]
    train_pred_subset = final_dict["train_subset_acc"]
    test_pred = final_dict["test_full_acc"]

    train_subset_cwa = test_pred
    
    
    flipped = defaultdict(dict)
    for key, val in train_subset_cwa.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    
    return flipped, combination


def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_c10_2_kl_torch_fairness_computation_95Per/"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"
    filenames = os.listdir(path_str)
    random.shuffle(filenames)
    # filenames = filenames[0:10]

    train_subset_cwas = {}

    train_subset_cwas_unmod = defaultdict(list)
    train_subset_cwas_zeroth = defaultdict(list)
    train_subset_cwas_first = defaultdict(list)
    train_subset_cwas_second = defaultdict(list)
    portions = []
    for filename in tqdm(filenames):
        train_subset_cwa, combination = get_kl_data(path_str + filename)
        portions = np.array(list(train_subset_cwa[0].keys())).astype(float)
        # portions = (10000 - portions) / 10000
        portions = 1 - portions

        # train_subset_cwas[combination] = train_subset_cwa
        for class_dictionary_key, class_dictionary in train_subset_cwa.items():
            if class_dictionary_key == combination[0]:
                train_subset_cwas_zeroth[class_dictionary_key].append(list(class_dictionary.values()))
            elif class_dictionary_key == combination[1]:
                train_subset_cwas_first[class_dictionary_key].append(list(class_dictionary.values()))
            elif class_dictionary_key == combination[2]:
                train_subset_cwas_second[class_dictionary_key].append(list(class_dictionary.values()))
            else:
                train_subset_cwas_unmod[class_dictionary_key].append(list(class_dictionary.values()))


    data_new = []
    for data in [train_subset_cwas_zeroth, train_subset_cwas_first, train_subset_cwas_second]:
        data = np.vstack(list(data.values()))
        data = pd.DataFrame(data, columns=[f'col{i}' for i in range(1, data.shape[1]+1)])
        data = pd.melt(data)
        data_new.append(data)

    label_list = ["Remove random", "Remove low LSI", "Remove high LSI"]
    for i, df in enumerate(data_new):
        df['type'] = label_list[i]

    data_new = pd.concat(data_new, ignore_index=True)
    
    portions = np.char.add(np.char.mod('%.1f', portions*100), '%')

    fig = plt.figure(figsize=(3.5, 2.5))
    # [none, random, first, last]
    p = sns.lineplot(data=data_new, y="value", x="variable", palette='viridis', hue="type")

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)
    plt.rc('axes',  titlesize=8)

    plt.xticks(np.arange(len(portions))[::2], portions[::2], rotation='vertical')
    # plt.ylim([0, 0.8])
    plt.xlabel("Portion of the Intial Sample Count")
    plt.ylabel("Train Accuracy")
    sns.move_legend(p, "lower left", frameon=False, title=None)
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = save_dir + "fair_train_ci"
    plt.savefig(save_name + ".png", format="png")
    print(f"saving fig as {save_name}.png")



    
main()