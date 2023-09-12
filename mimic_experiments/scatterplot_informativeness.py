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
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True)
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



def main():
    tar_label = 8
    path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results/gradients"
    data_list = read_pkl_files_in_folder(path)
    data_list = limit_to_final(data_list)
    idx, kl1, kl2 = get_idx_kl1_kl2(data_list)

    images, label = get_images_form_idx(idx)
    images, idx, kl1, kl2, label = limit_to_class(images, idx, kl1, kl2, label, tar_label)
    print(f"Remaining idx: {len(idx)}")
    fig, ax = plt.subplots()
    imscatter(kl1, kl2, images, zoom=0.1, ax=ax)
    ax.set_xlabel(r'$KL(D^{-1}||D)$')
    ax.set_ylabel(r'$KL(D||D^{-1})$')
    plt.savefig("output.jpg")

main()