import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from Datasets.dataset_helper import get_dataset
import cv2
plt.rcParams.update({
    'font.family': 'serif',
})

def read_pkl_files_in_folder(folder_path):
    data_list = []  # List to store the content of pkl files

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return data_list

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    data_list.append(data)
                    print(f"Loaded data from '{filename}'")
            except Exception as e:
                print(f"Error loading data from '{filename}': {str(e)}")

    return data_list

def limit_to_final(data):
    final_epoch_data = []
    for each in data:
        final_epoch_data.append(each[-1])
    return final_epoch_data

def get_idx_kl1_kl2(data):
    idx = []
    kl1 = []
    kl2 = []
    for each in data:
        idx.append(each["removed_idx"])
        kl1.append(each["mean_KL_divergence_model_vs_compare"])
        kl2.append(each["mean_KL_divergence_compare_vs_model"])
    return idx, kl1, kl2
    
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


def limit_to_class(images, idx, kl1, kl2, label, tar_label):
    images_out = []
    idx_out = []
    kl1_out = []
    kl2_out = []
    label_out = []
    
    for (im, id, k1, k2, lab) in zip(images, idx, kl1, kl2, label):
        if lab == tar_label:
            images_out.append(im)
            idx_out.append(id)
            kl1_out.append(k1)
            kl2_out.append(k2)
            label_out.append(lab)
    return images_out, idx_out, kl1_out, kl2_out, label_out


def load_c_scores(path):
    c_data = np.load(path)
    labels = c_data["labels"]
    scores = c_data["scores"]
    return labels, scores

def find_common_indices_and_data(list1_indices, list2_indices, list1_data, list2_data):
    # Find the common indices between list1_indices and list2_indices
    common_indices = list(set(list1_indices).intersection(list2_indices))

    # Create lists to store the common data from list1_data and list2_data
    common_data_list1 = []
    common_data_list2 = []

    # Extract the common data based on the common indices
    for index in common_indices:
        common_data_list1.append(list1_data[list1_indices.index(index)])
        common_data_list2.append(list2_data[list2_indices.index(index)])

    return common_indices, common_data_list1, common_data_list2


def main():
    tar_label = 0
    plot = "mem_vs_KL"
    path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp3/gradients"
    path_to_c_scores = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz"
    data_list = read_pkl_files_in_folder(path)
    data_list = limit_to_final(data_list)
    idx, kl1, kl2 = get_idx_kl1_kl2(data_list)
    images, label = get_images_form_idx(idx)
    if plot =="KL_vs_KL":
        images, idx, kl1, kl2, label = limit_to_class(images, idx, kl1, kl2, label, tar_label)
        print(f"Remaining idx: {len(idx)}")
        fig, ax = plt.subplots()
        fig.set_dpi(500)
        fig.set_size_inches(10, 10)
        imscatter(kl1, kl2, images, zoom=0.5, ax=ax)
        ax.set_xlabel(r'$KL(D^{-1}||D)$')
        ax.set_ylabel(r'$KL(D||D^{-1})$')
        plt.savefig("output.jpg")
    elif plot == "c_vs_KL":
        kl_max = np.maximum(kl1, kl2)
        c_labels, c_scores = load_c_scores(path_to_c_scores)
        c_scores = c_scores[idx]
        c_labels = c_labels[idx]
        test = c_labels == label
        print(all(test))
        images, idx, kl1, c_scores, label = limit_to_class(images, idx, kl_max, c_scores, label, tar_label)
        print(f"Remaining idx: {len(idx)}")

        kl1_norm = [(kl1_id-min(kl1))/ (max(kl1) - min(kl1)) for kl1_id in kl1]
        color = [abs(kl1_id - c_score_id) for (kl1_id, c_score_id) in zip(kl1_norm, c_scores)]
        color = [[c, 0, 0] for c in color]

        # color = []
        # for ai in idx:
        #     if ai < 500:
        #         color.append([1, 0, 0])
        #     elif ai < 1000: 
        #         color.append([0, 1, 0])
        #     elif ai < 1500:
        #         color.append([0, 0, 1])
        #     else:
        #         color.append([0, 0, 0])
        fig, ax = plt.subplots()
        fig.set_dpi(1000)
        fig.set_size_inches(10, 10)
        # imscatter(kl1, c_scores, images, zoom=0.5, ax=ax)
        plt.scatter(kl1, c_scores, c=color)
        ax.set_xlabel(r'$max(KL(D^{-1}||D), KL(D||D^{-1}))$')
        ax.set_ylabel(r'C-Score')
        plt.savefig("output_c_score.jpg")
    elif plot == "KL1_vs_KL2":
        images, idx, kl1, kl2, label = limit_to_class(images, idx, kl1, kl2, label, tar_label)
        path2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp/gradients"
        data_list2 = read_pkl_files_in_folder(path2)
        data_list2 = limit_to_final(data_list2)
        idx2, kl12, kl22 = get_idx_kl1_kl2(data_list2)
        common_indices, common_data_list1, common_data_list2 = find_common_indices_and_data(idx, idx2, kl1, kl12)
        fig, ax = plt.subplots()
        fig.set_dpi(500)
        fig.set_size_inches(10, 10)
        plt.scatter(common_data_list1, common_data_list2)
        ax.set_xlabel(r'$KL(D^{-1}||D)_mlp_high_lr$')
        ax.set_ylabel(r'KL(D^{-1}||D)_mlp_low_lr')
        plt.savefig("output_lr_compare.jpg")
    elif plot == "mem_vs_KL":
        path2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feldman/mlp_cifar10_150_1000.pkl"
        with open(path2, 'rb') as file:
            mem_data = pickle.load(file)
        mem_data = mem_data["data"]
        mem_list = []
        for index in idx:
            mem_list.append(mem_data.get(index, 0))
        fig, ax = plt.subplots()
        fig.set_dpi(500)
        fig.set_size_inches(10, 10)
        plt.scatter(mem_list, kl1)
        ax.set_xlabel(r'$mem_data$')
        ax.set_ylabel(r'KL(D^{-1}||D)')
        plt.savefig("output_mem_kl_compare.jpg")

main()