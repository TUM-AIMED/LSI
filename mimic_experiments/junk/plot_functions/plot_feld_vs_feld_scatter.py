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

def get_feld_scores(feld_path, idx):
    with open(feld_path, 'rb') as file:
        data = pickle.load(file)
    memorization = data["memorization"]
    influence = data["influence"]
    indexed_memorization = [memorization[index] for index in idx]
    indexed_influence = [influence[index] for index in idx]
    return indexed_memorization, indexed_influence

def main():
    feld_path1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld/feldman_cifar10_first1500_final_1200.pkl"
    feld_path2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld/test_feldman_cifar10_1200.pkl"
    file_name = "z_feld_vs_feld_scatter_cifar"

    feld_scores1, _ = get_feld_scores(feld_path1, [*range(1200)])
    feld_scores2, _ = get_feld_scores(feld_path2, [*range(1200)])


    fig, ax = plt.subplots(1,figsize=(6,6))
    plt.scatter(feld_scores1, feld_scores2, marker='o', color='blue')
    plt.xlabel("Mem-scores - part")
    plt.ylabel("Mem-scores - full")

    ax.plot([0, np.max(feld_scores1) + 0.1], [0, np.max(feld_scores2) + 0.1], linestyle='--', color='blue')
    #use add_patch instead, it's more clear what you are doing
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as {file_name}.jpg")
main()