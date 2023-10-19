import os
import io
import pickle
import torch
import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib import animation
from Datasets.dataset_helper import get_dataset
import cv2
from collections import defaultdict


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

def normalize_to_0_1(data):
    if not data:
        return []  # Return an empty list if the input is empty

    min_value = min(data)
    max_value = max(data)

    normalized_data = [(x - min_value) / (max_value - min_value) for x in data]

    return normalized_data


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


def get_indiv_grads(means_path, idx, portion=None):
    means_list = read_pkl_files_in_folder(means_path)
    means_list = [run_data[-1] for run_data in means_list]

    data_list = means_list
    # Step 1: Initialize a defaultdict to collect lists for each index
    result_dict = defaultdict(list)

    # Step 2 and 3: Process the input dictionary
    for inner_dict in data_list:
        values = inner_dict['idp_accountant_indiv']
        removed_index = inner_dict['removed_idx']
        
        # Replace the removed index with None
        modified_values = values[:removed_index] + [None] + values[removed_index:]
        
        # Append the modified list to the defaultdict
        result_dict[removed_index].append(modified_values)
    
    if portion != None:
        result_dict = {key: value for key, value in result_dict.items() if portion[0] <= key <= portion[1]}
    # Step 4: Transpose the lists
    transposed_lists = list(result_dict.values())
    transposed_lists = [double_list[0] for double_list in transposed_lists]
    # Step 5: Calculate the average for each index
    transposed_lists2 = []
    for run in transposed_lists:
        run = [run[index] for index in idx]
        transposed_lists2.append(run)
    transposed_lists = transposed_lists2
    results_list = []
    for i in tqdm(range(len(transposed_lists[0]))):
        data_point_results = []
        for run in transposed_lists:
            data_point_results.append(run[i])
        num_none = data_point_results.count(None)
        data_point_results = [data_point for data_point in data_point_results if data_point != None]
        data_point_results = np.array(data_point_results)
        data_point_results_mean = np.mean(data_point_results, axis=0)
        results_list.append(data_point_results_mean)


    return results_list 

def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_replace_cifar_logged_grads2500_idx_0_kl/final.pkl"
    means_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_replace_cifar_logged_grads2500_idx_0"


    kl_data_max, idx = get_kl_data(kl_path, "kl2_max")
    grad_norms_epochwise = get_indiv_grads(means_path, idx)
    images, label = get_images_form_idx(idx)

    n = 20
    largest_indices = sorted(range(len(kl_data_max)), key=lambda i: kl_data_max[i], reverse=True)[:n]
    smallest_indices = sorted(range(len(kl_data_max)), key=lambda i: kl_data_max[i], reverse=True)[-n:]

    indices = [*largest_indices, *smallest_indices]

    kl_data = [kl_data_max[index] for index in indices]
    grad_norms_epochwise = [grad_norms_epochwise[index] for index in indices]
    kl_data_norm = normalize_to_0_1(kl_data)

    file_name = "z_grad_vs_kl_line"
    
    fig, ax = plt.subplots()

    fig.set_dpi(500)
    fig.set_size_inches(10, 10)
    for data, color_value, kl in zip(grad_norms_epochwise, kl_data_norm, kl_data):
        x = np.arange(len(data))
        y = data
        color = plt.cm.viridis(color_value)  # Map the color value to a color using the viridis colormap
        ax.plot(x, y, label=f"{kl:.2f}", color=color)

    # Add labels, legend, and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.set_title('Line Plots with Different Colors')
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as {file_name}.jpg")

main()