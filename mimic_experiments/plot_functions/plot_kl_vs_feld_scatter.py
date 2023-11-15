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

def get_kl_data(final_path, agg_type, rand=False):
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
    if rand: 
        return kl_data_list, idx, final_dict["random_labels_idx"]
    return kl_data_list, idx, None


def get_feld_scores(feld_path, idx):
    with open(feld_path, 'rb') as file:
        data = np.load(file)
        memorization = data["tr_mem"]
    indexed_memorization = [memorization[index] for index in idx]
    return indexed_memorization

def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar100compressed_logreg4_3_1000__30_1000_0.9850convex_SGD_Plane_vs_Lion/results_all.pkl"
    feld_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Feld_Score/cifar100_infl_matrix.npz"
    file_name = "z_kl_vs_feld_scatter_cifar100compressed_logreg4_3_1000__30_1000_0.9850convex_SGD_Plane_vs_Lion"
    kl_diag1, idx, _ = get_kl_data(kl_path, "kl2_diag")
    kl_diag1 = [np.mean(data) for data in kl_diag1]
    combined_data = list(zip(kl_diag1, idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl_diag1, idx = zip(*sorted_data)

    first_10 = kl_diag1[:10]
    last_10 = kl_diag1[-10:]
    kl_diag1 = first_10 + last_10

    first_10 = idx[:10]
    last_10 = idx[-10:]
    idx = first_10 + last_10

    feld_scores = get_feld_scores(feld_path, idx)


    images, label = get_images_form_idx(idx)


    fig, ax = plt.subplots(1,figsize=(6,6))
    # plt.scatter(kl_diag1, feld_scores, marker='o', color='blue')
    imscatter(kl_diag1, feld_scores, images, ax=None, zoom=1)
    plt.xlabel("kl_diag2 - 200")
    plt.ylabel("Mem-scores")
 

    # for i, (x, y, x_var, y_var) in enumerate(zip(kl_diag1, kl_diag2, kl_var1, kl_var2)):
        # if i %  50 == 0:
        #     ellipse = Ellipse((x, y), width=x_var, height=y_var, edgecolor='red', facecolor='red', alpha=0.5)
        #     ax.add_patch(ellipse)
    # ax.plot([0, np.max(kl_diag1) + 2], [0, np.max(feld_scores) + 0.1], linestyle='--', color='blue')
    #use add_patch instead, it's more clear what you are doing
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")
main()