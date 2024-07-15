import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from Datasets.dataset_helper import get_dataset
from collections import defaultdict


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



def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar10")
    data_set = dataset_class(data_path, train=True)
    data_set._set_classes([0, 5])
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



def get_color(images):
    colors = []
    for image in images:
        colors.append(np.mean(image))
    return colors

def get_kl_data(final_path):
    with open(final_path, 'rb') as file:
        final_list = pickle.load(file)
    final_list = final_list["kl1_max"]
    key_list = list(final_list.keys())
    key_list.sort()
    idx = key_list
    kl_data = [final_list[key] for key in key_list]
    return kl_data, idx

def get_rdp(means_path, idx, portion=None):
    means_list = read_pkl_files_in_folder(means_path)
    means_list = [run_data[-1] for run_data in means_list]

    data_list = means_list
    # Step 1: Initialize a defaultdict to collect lists for each index
    result_dict = defaultdict(list)

    # Step 2 and 3: Process the input dictionary
    for inner_dict in data_list:
        values = inner_dict['idp_accountant']
        removed_index = inner_dict['removed_idx']
        
        # Replace the removed index with None
        modified_values = values[:removed_index] + [None] + values[removed_index+1:]
        
        # Append the modified list to the defaultdict
        result_dict[removed_index].append(modified_values)
    
    if portion != None:
        result_dict = {key: value for key, value in result_dict.items() if portion[0] <= key <= portion[1]}
    # Step 4: Transpose the lists
    transposed_lists = list(result_dict.values())
    transposed_lists = [double_list[0] for double_list in transposed_lists]
    # Step 5: Calculate the average for each index
    averages = [sum(filter(None, col)) / (len(col) - col.count(None)) for col in zip(*transposed_lists)]
    indexed_averages = [averages[index] for index in idx]
    return indexed_averages


def main():
    final_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace_cifar390_plane_vs_dog/final/final.pkl"
    means_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace_cifar390_plane_vs_dog/means"

    kl_data, idx = get_kl_data(final_path)
    averages = get_rdp(means_path, idx)
    images, labels = get_images_form_idx(idx)

    mask = [i for i in range(len(labels)) if labels[i] == 1]

    kl_data = [kl_data[i] for i in mask]
    images = [images[i] for i in mask]
    labels = [labels[i] for i in mask]
    idx = [idx[i] for i in mask]

    best_indices = np.argsort(kl_data)[:50]
    worst_indices = np.argsort(kl_data)[-50:]


    fig, axs = plt.subplots(4, 25, figsize=(40, 10))

    # Plot the images from lowest to highest values
    for i, idx in enumerate(best_indices):
        ax = axs[i // 25, i % 25]
        new_array = images[idx]
        ax.imshow(new_array)
        kl_value = kl_data[idx]
        label = labels[idx]
        ax.set_title(f"KL: {kl_value:.2f}\nLabel: {label:.2f}")
        ax.axis("off")

    for i, idx in enumerate(worst_indices):
        ax = axs[2 + i // 25, i % 25]
        new_array = images[idx]
        ax.imshow(new_array)
        kl_value = kl_data[idx]
        label = labels[idx]
        ax.set_title(f"KL: {kl_value:.2f}\nLabel: {label:.2f}")
        ax.axis("off")

    # Save or show the figure.
    plt.savefig('pearl_necklace2.png')  # Save the figure as an image.

main()