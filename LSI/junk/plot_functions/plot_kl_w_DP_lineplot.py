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
from matplotlib.colors import TABLEAU_COLORS
from sklearn.preprocessing import LabelBinarizer

def one_hot_encoding(labels):
    lb = LabelBinarizer()
    one_hot_encoded = lb.fit_transform(labels)
    return lb.classes_, one_hot_encoded.tolist()


plt.rcParams.update({
    'font.family': 'serif',
})

def get_n_distinct_colors(n):
    # Get a list of distinct colors from the TABLEAU_COLORS colormap
    if n > 10:
        return ['#e377c2' for i in range(n)]
    all_colors = list(TABLEAU_COLORS.values())
    distinct_colors = list(TABLEAU_COLORS.values())[:n]
    return distinct_colors

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


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
    dp_values = list(kl_data[0].keys())
    idx = list(kl_data[0][list(kl_data[0].keys())[0]].keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data_dict = defaultdict(lambda: defaultdict(list))
    for seed in kl_data:
        for priv in kl_data[seed]:
            for index in idx:
                kl_data_dict[priv][index].append(kl_data[seed][priv][index])
    for priv, priv_wise in kl_data_dict.items():
        for index, item_wise in priv_wise.items():
            kl_data_dict[priv][index] = np.mean(kl_data_dict[priv][index])
    if rand: 
        return kl_data_dict, idx, dp_values
    return kl_data_dict, idx, dp_values

def get_images_form_idx(idxs, dataset_name):
    dataset_class, data_path = get_dataset(dataset_name)
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    labels = data_set.labels[idxs]
    images = data_set.data[idxs]
    images = [im.transpose(1, 2, 0) for im in images]
    return images, labels


def main():
    dataset_name = "cifar10"
    compare1 = 2
    compare2 = 0
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_w_DP_final/results_cifar10compressed_logreg4_DP_Range_m09_1_5_30_0.5_100_50000_0.5494_3.0/results_all.pkl"
    file_name = f"z_kl_w_DP_lineplot_final_cifar10compressed_logreg4_DP_Range_m09_1_5_30_0.5_100_50000_0.5494_3.0_{compare1}_{compare2}_sorted_by_com2_diff_seed"
    
    rand = True
    kl1_diag, init_idx, dp_values = get_kl_data(kl_path, "kl1_diag", rand=rand)
    result_list = [
        [
            np.mean(kl1_diag[key][subkey])
            for subkey in kl1_diag[key]
        ]
        for key in kl1_diag
    ]
    kl1_diag = np.transpose(np.array(result_list))
    combined_data = zip(kl1_diag, init_idx)
    sorted_list = sorted(combined_data, key=lambda x: x[0][compare2])
    # kl1_diag, idx = zip(*sorted_data)
    (kl_data, idx) = list(zip(*sorted_list))
    images, labels = get_images_form_idx(list(idx), dataset_name)
    test = np.unique(labels)

    label_ass, label_1hot = one_hot_encoding(labels)


    colors = get_n_distinct_colors(len(np.unique(labels)))
    mapping_dict = dict(zip(set(labels), colors))
    labels2 = [mapping_dict[value] if value in mapping_dict else value for value in labels]

    kl_data = np.array(kl_data)
    kl_data = kl_data.transpose()
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    for i, (data, dp_label) in enumerate(zip(kl_data, dp_values)): # enumerate(zip([kl_data[compare1], kl_data[compare2]], [dp_values[compare1], dp_values[compare2]])):
        plt.plot(list(range(len(data))), data, marker='o', linestyle='-', color=(0, 0.1 + 0.3*i, 0.1 + 0.3*i), label=dp_label)
    plt.scatter(list(range(len(data))), [-0.05 for i in range(len(data))], c = labels2)
    # Show legend
    plt.legend(title="DP - sigma")
    plt.xlabel("Index")
    plt.ylabel("KL1 - Diag")
    plt.xticks(list(range(len(idx))), idx)
    # plt.ylim([0, 1])

    plt.subplot(1, 2, 2)
    for (label, label_name) in zip(np.transpose(label_1hot), label_ass):
        plt.plot(list(range(len(label))), np.cumsum(label)/np.sum(label), label=label_name)
    plt.xticks(list(range(len(idx))), idx)  
    plt.legend(title="Labels")
  
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

main()