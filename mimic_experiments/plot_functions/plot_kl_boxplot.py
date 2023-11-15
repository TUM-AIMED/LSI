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
    dataset_class, data_path = get_dataset("mnist")
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

def get_correct(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    pred_data = final_dict["correct_data"]
    idx = list(pred_data[0].keys())
    pred_data_dict = defaultdict(list)
    for seed in pred_data:
        for index in idx:
            pred_data_dict[index].append(np.concatenate(pred_data[seed][index]))
    predict_own = {}
    for index, index_data in pred_data_dict.items():
        correct = 0
        for run_data in index_data:
            test = run_data[idx.index(index)]
            if run_data[idx.index(index)] == True:
                correct += 1
            else:
                continue
        predict_own[index] = correct
    predict_own_list = [value/len(pred_data) for key, value in predict_own.items()]
    all_pred = []
    for index, index_data in pred_data_dict.items():
        for run_data in index_data:
            all_pred.append(run_data)
    all_pred = np.stack(all_pred)
    count_per_column = np.sum(all_pred, axis=0)
    count_per_column = count_per_column/all_pred.shape[0]
    return predict_own_list, count_per_column


def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar100compressed_logreg4_2_400__1000_50000_0.5969convex_SGD/results_all.pkl"
    file_name = "z_boxplot_cifar100compressed_logreg4_2_400__1000_50000_0.5969convex_SGD"
    rand = True
    kl1_diag, init_idx, rand_labels_idx = get_kl_data(kl_path, "kl2_diag", rand=rand)
    kl1_diag = np.abs(kl1_diag)
    predict_own_list, count_per_column = get_correct(final_path=kl_path)
    # predicted_own = 1 if predicted in all tasks even though excluded
    # count_per_col = 1 if predicted in all tasks
    combined_data = list(zip(kl1_diag, init_idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl1_diag, idx = zip(*sorted_data)

    plt.figure(figsize=(50, 6))
    plt.boxplot(kl1_diag, labels=idx)
    plt.xlabel("Index")
    plt.ylabel("KL1 - Diag")

    if rand:
        for true_idx in rand_labels_idx:
            if true_idx > max(idx):
                continue
            plot_index = list(idx).index(true_idx)
            plt.scatter(plot_index + 1, 0.0, color='blue')
    for true_idx in idx:
        plot_index = list(idx).index(true_idx)
        plt.scatter(plot_index + 1, -0.5, color=str(predict_own_list[init_idx.index(true_idx)]),edgecolors='black') # the more white the more often predicted when excluded
        plt.scatter(plot_index + 1, -0.9, color=str(count_per_column[init_idx.index(true_idx)]),edgecolors='black') # the more white the more often predicted overall


    plt.tight_layout()
    plt.show()
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

main()