import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from Datasets.dataset_helper import get_dataset
import cv2
plt.rcParams.update({
    'font.family': 'serif',
})

def read_pkl_files_in_folder(folder_path):
    data_list = []  # List to store the content of pkl files

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return data_list

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    data_list.append(data)
                    print(f"Loaded data from '{filename}'")
            except Exception as e:
                print(f"Error loading data from '{filename}': {str(e)}")

    return data_list

def limit_to_final(data):
    final_epoch_data = []
    for each in data:
        final_epoch_data.append(each[-1])
    return final_epoch_data

def get_idx_kl1_kl2(data):
    idx = []
    kl1 = []
    kl2 = []
    for each in data:
        idx.append(each["removed_idx"])
        kl1.append(each["mean_KL_divergence_model_vs_compare"])
        kl2.append(each["mean_KL_divergence_compare_vs_model"])
    return idx, kl1, kl2

def limit_to_class(data, label, tar_label):
    results = []
    for each in data:
        results_each = []
        for (datapoint, lab) in zip(each, label):
            if lab == tar_label:
                results_each.append(datapoint)
        results.append(results_each)
    return results



path_wo_reorder = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp3/gradients"


data_list = read_pkl_files_in_folder(path_wo_reorder)
data_list = limit_to_final(data_list)
idx, kl1, kl2 = get_idx_kl1_kl2(data_list)

    # Create the bar plot
plt.figure()
plt.hist(kl1, bins=25)
plt.title('Weight Distribution Histogram')
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.grid(True)
plt.savefig("kl_hist" + ".jpg")
print(f"saved fig at {'kl_hist' + '.jpg'}")
