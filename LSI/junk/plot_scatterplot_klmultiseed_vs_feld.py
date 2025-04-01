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
    idx_to_kllist_dict = defaultdict(list)
    for seed in kl_data:
        for idx, kl_value in kl_data[seed].items():
            idx_to_kllist_dict[idx].append(kl_value)
    results_idx = []
    results_kl = []
    results_kl_all = []
    results_std = []
    for idx, kl_list in idx_to_kllist_dict.items():
        results_idx.append(idx)
        results_kl_all.append(kl_list)
        results_kl.append(np.mean(np.array(kl_list)))
        results_std.append(np.std(np.array(kl_list)))
    return results_kl, results_kl_all, results_std, results_idx


def load_c_scores(path):
    c_data = np.load(path)
    labels = c_data["labels"]
    scores = c_data["scores"]
    return labels, scores


def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_diag_asdlgnn_10_300_test/results_all.pkl"
    c_score_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz"
    kl_path2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_diag_backpackef_150_30/results_all.pkl"

    kl_data, kl_all, kl_std, idx = get_kl_data(kl_path, "mean_diff_mean")
    # kl_data2, kl_all2, kl_std2, idx2 = get_kl_data(kl_path2, "kl2")
    images, label = get_images_form_idx(idx)

    clabels, cscores = load_c_scores(c_score_path)
    cscores = [cscores[index] for index in idx]
    clabels = [clabels[index] for index in idx]

    ordering = np.argsort(cscores)
    kl_data = [kl_data[index] for index in ordering]
    kl_all = [kl_all[index] for index in ordering]
    kl_std = [kl_std[index] for index in ordering]
    idx = [idx[index] for index in ordering]
    
    plt.boxplot(kl_all, labels=idx)

    # Set plot title and axis labels
    plt.title("Boxplot of Data")
    plt.xlabel("idx")
    plt.ylabel("weight_diff_mean")

    file_name = "z_kl_weight_diff_mean_vs_c_300"
    plt.savefig(file_name + "_boxplot" + ".jpg")
    print(f"saving fig as {file_name+ '_boxplot'}.jpg")


    xcompare = kl_data
    ycompare = cscores
    zcompare = None
    x_label = "weight_diff_sum"
    y_label = "c-scores"
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