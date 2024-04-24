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

def plot_multiline(kl_path):
    idx_all = get_kl_data(kl_path, "kl1_diag")
    results_all = {}
    for idx in np.unique(idx_all[0]):
        idx_wise_idx_list = []
        for remove_array in idx_all:
            if idx in remove_array:
                idx_wise_idx_list.append(remove_array.index(idx))
        results_all[idx] = idx_wise_idx_list

    data = list(results_all.values())
    idxs = list(results_all.keys())

    im, labels = get_images_form_idx(idxs)

    plt.figure(figsize=(16, 25), dpi=1000)
    for i, (idx, values_list, lab) in enumerate(zip(idxs, data, labels)):
        if lab == 81:
            color = (0.8, 0.0, i*0.001)
        elif lab == 45:
            color = (0.0, 0.8, i*0.001)
        else:
            color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(values_list, label=f'List {idx}', color=color)

    # Set labels and title
    plt.xlabel('Remove_count')
    plt.ylabel('Position')

    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")


def plot_imagestring(kl_path):
    idx = get_removed(kl_path)
    imgages, labels = get_images_form_idx(idx)
    fig, axes = plt.subplots(1, len(imgages), figsize=(200, 1))
    fontdict = {'fontsize': 2}
    for i, (img, lab) in enumerate(zip(imgages, labels)):
        axes[i].imshow(img)
        axes[i].set_title(lab, fontdict=fontdict)
        axes[i].axis('off')
    plt.savefig(file_name + "_string.jpg", dpi = 300)
    print(f"saving fig as ./{file_name}_string.jpg")

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
    for key, kl_data_seed in kl_data.items():
        idx_list = list(kl_data_seed.keys())
        value_list = list(kl_data_seed.values())
        combined_data = list(zip(value_list, idx_list))
        sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
        value_list, idx_list = zip(*sorted_data)
        idx_all.append(list(idx_list))
    return idx_all

def get_removed(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    diff_data = final_dict["difficult_idx"]
    all_idx = []
    for key, value in diff_data.items():
        all_idx.extend(value)
    return all_idx



kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_diff/results_cifar100compressed_logreg4_10_5_25_1000_0.9654_0.3/results_all.pkl"
file_name = "z_onioning"

plot_multiline(kl_path)
plot_imagestring(kl_path)
print("")