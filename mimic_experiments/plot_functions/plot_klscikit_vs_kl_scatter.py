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


plt.rcParams.update({
    'font.family': 'serif',
})

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def imscatter(x, y, images, ax=None, zoom=1):
    OffsetImages = []
    if ax is None:
        ax = plt.gca()
    for image in images:
        OffsetImages.append(OffsetImage(image, zoom=zoom))
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, OffsetImages):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar100")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    labels = data_set.labels[idxs]
    images = data_set.data[idxs]
    images = [im.transpose(1, 2, 0) for im in images]
    return images, labels

def get_kl_data_scikit(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    data_dict = final_dict[0]
    data = []
    idx = []
    for key, value in data_dict.items():
        idx.append(key)
        data.append(value)
    print("")
    return data, idx

def get_kl_data(final_path, agg_type):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict[agg_type]
    idx = list(kl_data[0].keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data_dict = defaultdict(list)
    for seed in kl_data:
        for index in idx:
            kl_data_dict[index].append(kl_data[seed][index])
    kl_data_list = [value for key, value in kl_data_dict.items()]
    return kl_data_list, idx

def find_common_items_with_indices(list1, list2):
    common_items = []
    indices_list1 = []
    indices_list2 = []

    for index1, item1 in enumerate(list1):
        for index2, item2 in enumerate(list2):
            if item1 == item2:
                common_items.append(item1)
                indices_list1.append(index1)
                indices_list2.append(index2)

    return common_items, indices_list1, indices_list2


def main():
    kl_path_scikit = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_scikit/results_all.pkl"
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar100compressed_logreg4_2_400__1000_50000_0.5969convex_SGD/results_all.pkl"

    file_name = "z_klscikit_vs_kl_scatter_cifar100compressed_logreg4_2_400__1000_50000_0.5969convex_SGD"
    kl_diag1_scikit, idx_scikit = get_kl_data_scikit(kl_path_scikit)
    kl_diag1, idx = get_kl_data(kl_path, "kl2_diag")
    kl_diag1 = [np.mean(data) for data in kl_diag1]
    common_items, indices_list1, indices_list2 = find_common_items_with_indices(idx_scikit, idx)
    kl_diag1_scikit = np.array(kl_diag1_scikit)
    kl_diag1 = np.array(kl_diag1)
    kl_diag1_scikit = kl_diag1_scikit[indices_list1]
    kl_diag1 = kl_diag1[indices_list2]

    # kl_diag1 = list(kl_diag1)
    # first_10 = kl_diag1[:50]
    # last_10 = kl_diag1[-20:]
    # kl_diag1 = list(first_10 + last_10)

    # first_10 = idx[:50]
    # last_10 = idx[-20:]
    # idx = list(first_10 + last_10)

    images, label = get_images_form_idx(idx)


    fig, ax = plt.subplots(1,figsize=(6,6))
    plt.scatter(kl_diag1, feld_scores, marker='o', color='blue')
    # imscatter(kl_diag1, feld_scores, images, ax=None, zoom=1)
    plt.xlabel("kl_diag2 - 200")
    plt.ylabel("Mem-scores")

    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")
main()