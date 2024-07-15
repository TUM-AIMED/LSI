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
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from collections import defaultdict


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


def get_grad_data(final_path):
    with open(final_path, 'rb') as file:
        grad_data = pickle.load(file)
    idxs = []
    result = []
    for seed, seed_data in grad_data.items():
        seed_wise_results = []
        idx = []
        for index, idx_data in seed_data.items():
            idx.append(index)
            seed_wise_results.append(np.sqrt(np.sum(idx_data)))
        result.append(seed_wise_results)
        idxs.append(idx)
    if not all(inner_list == idxs[0] for inner_list in idxs):
        raise Exception("some mix up")
    return np.array(idxs[0]), np.mean(result, axis=0)

def fifthorder(x, a, b, c, d, e, f, g):
    y = a*x**4 + b*x**4 + c*x**3 + d*x**2 + e*x + g
    return y

def log_fkt(x, a, b, c):
    y = c*np.log(x + a) - b
    return y


def main():
    grad_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_cifar100compressed_logreg4_1_100.0_1e-09__1000_50000/results.pkl"

    file_name = "z_label_vs_grad_boxplot_cifar100compressed_logreg4_1_100.0_1e-09__1000_50000"

    idx, grad_data = get_grad_data(grad_path)
    images, label = get_images_form_idx(idx)

    all_dict = defaultdict(list)
    for lab, data in zip(label, grad_data):
        all_dict[lab].append(data)

    all_list = []
    keys = []
    for key, value in all_dict.items():
        all_list.append(value)
        keys.append(key)
        
    combined_data = list(zip(all_list, keys))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    all_list, keys = zip(*sorted_data)

        
  
    fig, ax = plt.subplots(1,figsize=(30,8))
    plt.boxplot(all_list, labels=keys)
    # plt.scatter(label, grad_data, marker='o', color='blue')
    plt.xlabel("Label")
    plt.ylabel("Cummulative_Grad_Norm")
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")
main()