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



def get_kl_data(final_path, agg_type="", rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])[0]
    idx = np.array(final_dict["idx"])[0]
    corrupt_idx = np.array(final_dict["corrupt"])
    non_corrupt_idx = [index for index in idx if idx not in corrupt_idx]
    corrupt_kl = kl_data[corrupt_idx]
    non_corrupt_kl = kl_data[non_corrupt_idx]
    return corrupt_kl, non_corrupt_kl

def main():
    path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.05_including_predictions.pkl"
    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
    corrupt_kl, non_corrupt_kl = get_kl_data(path_str)
    kl_data = [corrupt_kl, non_corrupt_kl]

    label_list = ["Corrupted Data", "Uncorrupted Data"]
    long_df = pd.DataFrame()

    for i, data_array in enumerate(kl_data):
        temp_df = pd.DataFrame({'Value': data_array, 'Label': label_list[i]})
        long_df = pd.concat([long_df, temp_df])
    p = sns.kdeplot(data=long_df, log_scale=True, fill=True, alpha=0.5, hue="Label", x="Value", common_norm=False, linewidths=2, palette='viridis')
    plt.xlabel("LSI")
    plt.ylabel("Density")
    sns.move_legend(p, "upper right", frameon=False, title=None)
    plt.tight_layout()
    save_name = save_dir + "corruption"
    plt.savefig(save_name + ".pdf", format="pdf")
    print(f"saving fig as {save_name}.pdf")



    
main()