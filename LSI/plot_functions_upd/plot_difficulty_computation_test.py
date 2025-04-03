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
    test_accuracy_largest = np.array(list(final_dict["test_acc"]["ordering_largest"].values()))
    test_accuracy_largest_onion = np.array(list(final_dict["test_acc"]["ordering_largest_onion"].values()))
    test_accuracy_random = np.array(list(final_dict["test_acc"]["random_first_ordering"].values()))
    test_accuracy_largest_balanced = np.array(list(final_dict["test_acc"]["ordering_largest_balanced"].values()))
    return [test_accuracy_largest_balanced, test_accuracy_random]

def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_difficulty_computation/kl_jax_epochs_1000_remove_2_dataset_cifar10compressed_portions_20_lr_0.02_.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"

    test_accuracy_largest, test_accuracy_random = get_kl_data(path_str)
    # data = np.array([train_data, test_data])
    data = np.array([test_accuracy_largest])
    data = data[:,:,-1]
    data = data[:, 0:-1]
    color = [(1 - k / data.shape[1], 0, k / data.shape[1]) for k in range(data.shape[1])]
    print("")
    # Get the indices
    indices = np.arange(data.shape[1])

    # Create scatter plot
    plt.scatter(indices, data[0], color=color)

    # Label x-axis
    plt.xticks(indices, [f"" for i in range(data.shape[1])])

    # Set axis labels and title
    plt.xlabel('Subsets chosen according to the KL divergence')
    plt.ylabel('Test accuracy')


    # Add legend
    legend_labels = ['Low KL-div', 'Mid KL-div', 'High KL-div']
    legend_colors = [(0, 0, 1), (0.5, 0, 0.5), (1, 0, 0)]

    # Create proxy artists for legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]

    # Add legend with specified labels and colors
    plt.legend(legend_handles, legend_labels, loc='upper right')

    plt.tight_layout()
    save_name = save_dir + "difficulty_computation20_test"
    plt.savefig(save_name + ".pdf", format="pdf")
    print(f"saving fig as {save_name}.pdf")



    
main()