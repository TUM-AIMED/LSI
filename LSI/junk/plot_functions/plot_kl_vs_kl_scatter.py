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
    dataset_class, data_path = get_dataset("cifar10")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    for idx in idxs:
        image, label, _, _ = data_set.__getitem__(idx)
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        # image = data_set.data[idx, :, :, :],
        # image = cv2.resize(image[0], (3, 224, 224), interpolation=cv2.INTER_LINEAR)
        images.append(image)
        labels.append(label)
    return images, labels


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


def get_boxplot_like(data_list):
    meds = []
    iqrs = []
    for data in data_list:
        data.sort()

        # Calculate the median
        n = len(data)
        if n % 2 == 0:
            median = (data[n // 2 - 1] + data[n // 2]) / 2
        else:
            median = data[n // 2]
        q1 = data[n // 4]
        q3 = data[3 * n // 4]
        iqr = q3 - q1
        meds.append(median)
        iqrs.append(iqr)
    return meds, iqrs

def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar10_cnn_noisy_20_400/results_all.pkl"
    kl_path2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar10_cnn_20_400/results_all.pkl"
    file_name = "z_kl_vs_kl_scatter_cifar_200_vs_200_noisy"
    kl_diag1, idx = get_kl_data(kl_path, "kl1_diag")
    (kl_diag1, kl_var1) = get_boxplot_like(kl_diag1)
    kl_diag2, idx2 = get_kl_data(kl_path2, "kl1_diag")
    (kl_diag2, kl_var2) = get_boxplot_like(kl_diag2)
    overlap_idx = [value for value in idx if value in idx2]

    images, label = get_images_form_idx(idx)

    kl_diag1_keep = []
    for (kl_data, index) in zip(kl_diag1, idx):
        if index in overlap_idx:
            kl_diag1_keep.append(kl_data)
    kl_diag1 = kl_diag1_keep

    kl_diag2_keep = []
    for (kl_data, index) in zip(kl_diag2, idx):
        if index in overlap_idx:
            kl_diag2_keep.append(kl_data)
    kl_diag2 = kl_diag2_keep


    fig, ax = plt.subplots(1,figsize=(6,6))
    plt.scatter(kl_diag1, kl_diag2, marker='o', color='blue')
    plt.xlabel("kl_diag1 - 200")
    plt.ylabel("kl_diag1 - 10")
 

    # for i, (x, y, x_var, y_var) in enumerate(zip(kl_diag1, kl_diag2, kl_var1, kl_var2)):
        # if i %  50 == 0:
        #     ellipse = Ellipse((x, y), width=x_var, height=y_var, edgecolor='red', facecolor='red', alpha=0.5)
        #     ax.add_patch(ellipse)
    ax.plot([0, np.max(kl_diag1) + 2], [0, np.max(kl_diag2) + 2], linestyle='--', color='blue')
    #use add_patch instead, it's more clear what you are doing
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as {file_name}.jpg")
main()