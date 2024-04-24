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
    kl_data = np.array(final_dict["kl"])[0]
    idx = np.array(final_dict["idx"])[0]
    corrupt_idx = np.array(final_dict["corrupt"])
    pred = final_dict["pred"][0][0:kl_data.shape[0]]
    return kl_data, pred

def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_1000_dataset_cifar10compressed_subset_50000_corrupt_0.0_including_predictions.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
    kl, pred = get_kl_data(path_str)

    # Convert lists to numpy arrays for easier sorting
    float_values = np.array(kl)
    bool_values = np.array(pred)

    # Sort both arrays based on float_values
    sorted_indices = np.argsort(float_values)
    sorted_float_values = float_values[sorted_indices]
    sorted_bool_values = bool_values[sorted_indices]

    # Plot the sorted data
    plt.scatter(range(len(sorted_float_values)), sorted_float_values, c=np.where(sorted_bool_values, 'red', 'blue'))

    plt.tight_layout()
    save_name = save_dir + "corr"
    plt.savefig(save_name + ".pdf", format="pdf")
    print(f"saving fig as {save_name}.pdf")



    
main()