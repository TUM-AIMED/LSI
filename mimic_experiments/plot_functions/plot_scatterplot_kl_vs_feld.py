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


plt.rcParams.update({
    'font.family': 'serif',
})


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
    idx = list(kl_data.keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data = [kl_data[str(index)] for index in idx]
    return kl_data, idx


def get_feld_scores(feld_path, idx):
    with open(feld_path, 'rb') as file:
        data = pickle.load(file)
    memorization = data["memorization"]
    influence = data["influence"]
    indexed_memorization = [memorization[index] for index in idx]
    indexed_influence = [influence[index] for index in idx]
    return indexed_memorization, indexed_influence


def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_final_kl_all/final.pkl"
    feld_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld/feldman_cifar10_400.pkl"


    kl_data_max, idx = get_kl_data(kl_path, "kl2_max")
    images, label = get_images_form_idx(idx)

    mem, inf = get_feld_scores(feld_path, idx)

    tar_label = None
    plot = "test"

    file_name = "z_mem_vs_kl"
    xcompare = kl_data_max
    ycompare = mem
    zcompare = None
    x_label = "kl_data_max"
    y_label = "mem-score"
    show_images = False
    animate = True

    x_compare_data = xcompare
    y_compare_data = ycompare


    fig, ax = plt.subplots()
    fig.set_dpi(500)
    fig.set_size_inches(10, 10)
    if show_images:
        imscatter(x_compare_data[0:50], y_compare_data[0:50], images[0:50])
    else:
        plt.scatter(x_compare_data, y_compare_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as {file_name}.jpg")

main()