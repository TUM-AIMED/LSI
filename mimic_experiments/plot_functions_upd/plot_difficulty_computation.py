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
import pandas as pd
import seaborn as sns
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
    train_accuracy_largest = np.array(list(final_dict["train_acc_subset"]["ordering_largest"].values()))
    train_accuracy_random = np.array(list(final_dict["train_acc_subset"]["random_first_ordering"].values()))
    train_accuracy_largest_balanced = np.array(list(final_dict["train_acc_subset"]["ordering_largest_balanced"].values()))
    test_accuracy_largest = np.array(list(final_dict["test_acc"]["ordering_largest"].values()))
    test_accuracy_random = np.array(list(final_dict["test_acc"]["random_first_ordering"].values()))
    test_accuracy_largest_balanced = np.array(list(final_dict["test_acc"]["ordering_largest_balanced"].values()))
    single_class_train_acc = np.full((test_accuracy_largest_balanced.shape[0], test_accuracy_largest_balanced.shape[1]), final_dict["train_unique"].item())
    single_class_test_acc = np.full((test_accuracy_largest_balanced.shape[0], test_accuracy_largest_balanced.shape[1]), final_dict["test_unique"].item())
    
    train_accuracy_random = single_class_train_acc
    test_accuracy_random = single_class_train_acc
    # return [train_accuracy_largest, train_accuracy_largest_onion, train_accuracy_random, train_accuracy_largest_balanced], \
    #         [test_accuracy_largest, test_accuracy_largest_onion, test_accuracy_random, test_accuracy_largest_balanced], \
    #             ["largest", "largest_onioning", "random", "largest_balanced"]
    return [train_accuracy_largest_balanced, train_accuracy_random, single_class_train_acc], \
    [test_accuracy_largest_balanced, test_accuracy_random, single_class_test_acc],\
    ["Balanced subset by KL-div value", "Random Subset"], single_class_train_acc, single_class_test_acc

def main():
    dataset = "cifar10"
    model = "CNN"
    Kind = "Train"

    # datasets = ["cifar10", "cifar100", "Imagenette", "Imagewoof", "Prima_smaller"]
    # datasets = ["cifar10compressed", "cifar100compressed", "Imagenettecompressed", "Imagewoofcompressed", "Primacompressed"]

    # datasets = ["cifar100", "cifar10"]
    # datasets_names = ["CIFAR100", "CIFAR10"]
    # models = ["ResNet9"]
    # models_names = ["ResNet-9"]

    # datasets = ["cifar10", "cifar100", "Imagenette", "Imagewoof", "Prima_smaller"]
    # datasets = ["cifar10compressed", "cifar100compressed", "Imagenettecompressed", "Imagewoofcompressed", "Primacompressed"]
    # datasets_names = ["CIFAR10", "CIFAR100", "Imagenette", "Imagewoof", "Pneumonia"]
    # models = ["MLP", "CNN", "ResNet9", "ResNet18"]
    # models_names = ["MLP", "CNN", "ResNet-9", "ResNet-18"]
    models = ["Tinymodel"]
    models_names = ["Proxy"]
    datasets = ["cifar10compressed"]
    datasets_names = ["CIFAR10"]
    Kinds = ["Test", "Train"]

    for dataset, datasets_name in zip(datasets, datasets_names):
        for model, model_name in zip(models, models_names):
            for Kind in Kinds:
                try:
                    lr = 0.01
                    epochs = 500
                    if model == "ResNet9" or model == "ResNet18":
                        epochs = 400
                    if model == "ResNet9" or model == "Tinymodel":
                        lr = 0.004
                    base_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_torch_difficulty_computation_after_workshop/"
                    ext_str = "kl_jax_epochs_"+str(epochs)+"_remove_2_dataset_"+ dataset +"_model_"+ model +"_portions_3_lr_"+str(lr) +"_4orders_lrscheduler.pkl"
                    path_str = base_str + ext_str

                    save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_diff_new/"

                    train_data, test_data, labels, single_class_train_acc, single_class_test_acc = get_kl_data(path_str)

                    if Kind == "Test":
                        data = np.array([test_data])
                    elif Kind == "Train":
                        data = np.array([train_data])
                    horizontal_line = single_class_test_acc

                    label_for_legend = ["highest KL", "mid KL", "lowest KL"]

                    data_restruct = []
                    for i in range(data.shape[1]):
                        data_indiv = data[0, i, :, :500]
                        data_indiv =  data_indiv.squeeze()
                        df_wide = pd.DataFrame(data_indiv.T, columns=[i for i in range(data_indiv.shape[0])])
                        df_wide['epoch'] = df_wide.index
                        df_long = pd.melt(df_wide, id_vars=['epoch'], var_name='col', value_name='accuracy')
                        if i == 0:
                            def categorize_lsi(value):
                                if value in range(0, 1):
                                    return 'High LSI'
                                elif value in range(1, 2):
                                    return 'Mid LSI'
                                elif value in range(2, 3):
                                    return 'Low LSI'
                                else:
                                    return 'Unknown'

                            # Apply the categorization function to create a new column
                            df_long['LSI Category'] = df_long['col'].apply(categorize_lsi)
                        # elif i == 1: 
                        #     def categorize_lsi(value):
                        #         return "Random"
                        #     # Apply the categorization function to create a new column
                        #     df_long['LSI Category'] = df_long['col'].apply(categorize_lsi)
                        elif i == 2: 
                            def categorize_lsi(value):
                                return "Dummy Baseline"
                            # Apply the categorization function to create a new column
                            df_long['LSI Category'] = df_long['col'].apply(categorize_lsi)
                        data_restruct.append(df_long)
                        print("")

                    unified_df = pd.concat([data_restruct[0], data_restruct[1]], ignore_index=True)
                    # Iterate through each subplot
                    p = []
                    plt.figure(figsize=(3.5, 2.5))
                    show_legend = True
                    if Kind == "Test":
                        show_legend = False
                    p.append(sns.lineplot(data=unified_df, y="accuracy", x="epoch", palette='viridis', hue="LSI Category", legend=show_legend))
                    p.append(sns.lineplot(data=data_restruct[2], y="accuracy", x="epoch", palette=['k'], hue="LSI Category", linestyle='--', legend=show_legend))



                    # Create proxy artists for legend

                    # Add legend with specified labels and colors

                    # Set row titles
                    plt.rc('xtick', labelsize=8)
                    plt.rc('ytick', labelsize=8)
                    plt.rc('axes',  labelsize=8)
                    plt.rc('axes',  titlesize=8)

                    plt.ylabel(str(Kind) + ' Accuracy')
                    plt.xlabel('Epochs')
                    plt.ylim((0, 1))
                    if Kind == "Train":
                        plt.title(f"{datasets_name} - {model_name}")
                        for plot in p:
                            sns.move_legend(plot, "lower right", frameon=False, title=None, bbox_to_anchor=(0.99, -0.06)) # y, x von oben links 0.99, -0.06   1, 0.53

                    else:
                        plt.title("  ")
                    for plot in p:
                        xticks = plot.get_xticks()
                        new_xticks = xticks * 10
                        new_xticks = [int(tick) for tick in new_xticks]

                        # Set the new x-tick labels
                        plot.set_xticklabels(new_xticks)


                    # Adjust layout
                    plt.tight_layout()
                    save_name = save_dir + "difficulty_computation3_"+str(dataset)+"_"+str(model)+"_lrschedule_" + str(Kind)
                    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
                    print(f"saving fig as {save_name}.pdf")
                    plt.savefig(save_name + ".png", format="png", dpi=1000)
                    print(f"saving fig as {save_name}.png")
                except:
                    print(f"Not found: {ext_str}")


main()