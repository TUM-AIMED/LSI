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
from matplotlib.patches import Ellipse, Circle
import random

def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar100")
    data_set = dataset_class(data_path, train=True)
    # data_set.reduce_to_active_class([81, 45])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    for idx in idxs:
        image, label, _, _ = data_set.__getitem__(idx, normalize=False)
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        # image = data_set.data[idx, :, :, :],
        # image = cv2.resize(image[0], (3, 224, 224), interpolation=cv2.INTER_LINEAR)
        images.append(image)
        labels.append(label)
    return images, labels

def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict[agg_type]
    idx_all = []
    sorted_values = []
    for key, kl_data_seed in kl_data.items():
        idx_list = list(kl_data_seed.keys())
        value_list = list(kl_data_seed.values())
        combined_data = list(zip(value_list, idx_list))
        sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
        value_list, idx_list = zip(*sorted_data)
        idx_all.append(list(idx_list))
        sorted_values.append(value_list)
    return idx_all, sorted_values

def get_removed(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict["kl1_diag"]
    idx = list(kl_data.keys())
    return idx



kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_infl/results_cifar100compressed_logreg4_5_25_1000_0.9880_0.341_39/results_all.pkl"
file_name = "z_influence_kl_"

idx_order, kl_data_order = get_kl_data(kl_path, "kl1_diag")
idx_rem = get_removed(kl_path)
idx_rem = idx_rem[1:]
idx_rem = [[idx_rem_indiv] for idx_rem_indiv in idx_rem]

kl_change = []
idx_change = []
for i in range(len(kl_data_order) - 1):
    kl_change_step = []
    idx_change_step = []
    for idx in np.unique(idx_order[i+1]):
        idx_change_step.append(idx)
        kl_change_step.append((kl_data_order[i+1][idx_order[i+1].index(idx)] - kl_data_order[0][idx_order[0].index(idx)]))
        # kl_change_step.append((idx_order[i+1].index(idx) - idx_order[0].index(idx)))

    combined_data = list(zip(kl_change_step, idx_change_step))
    sorted_data = sorted(combined_data, key=lambda x: x[0])
    kl_change_step, idx_change_step = zip(*sorted_data)
    kl_change.append(list(kl_change_step))
    idx_change.append(list(idx_change_step))

most_influenced = []
most_influenced_kl = []
for run_data, run_data_kl in zip(idx_change, kl_change):
    most_influenced.append(run_data[-5:])
    most_influenced_kl.append(run_data_kl[-5:])
    # most_influenced.append(run_data[0:5])
    # most_influenced_kl.append(run_data_kl[0:5])

num_rows = len(most_influenced)
num_cols = len(most_influenced[0]) + len(idx_rem[0]) + 1
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 1 * num_rows)) 

for i, run_data in enumerate(idx_rem):
    img, _ = get_images_form_idx(run_data)
    for j, image in enumerate(img):
        axs[i, j].imshow(image)
        axs[i, j].axis("off")
        axs[i, j + 1].axis("off")

for i, (run_data, run_data_kl) in enumerate(zip(most_influenced, most_influenced_kl)):
    img, _ = get_images_form_idx(run_data)
    for j, (image, kl_change) in enumerate(zip(img, run_data_kl)):
        axs[i, j + len(idx_rem[0]) + 1].imshow(image)
        axs[i, j + len(idx_rem[0]) + 1].axis("off")
        axs[i, j + len(idx_rem[0]) + 1].text(0.5, 1.05, kl_change, ha='center', va='center', transform=axs[i, j + len(idx_rem[0]) + 1].transAxes, fontsize=12, color='blue')

plt.savefig(file_name + "_string.jpg", dpi = 300)
print(f"saving fig as ./{file_name}_string.jpg")
print("")