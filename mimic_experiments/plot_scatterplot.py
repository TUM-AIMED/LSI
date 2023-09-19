import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib import animation
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


def limit_to_class(data, label, tar_label):
    results = []
    for each in data:
        results_each = []
        for (datapoint, lab) in zip(each, label):
            if lab == tar_label:
                results_each.append(datapoint)
        results.append(results_each)
    return results


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
    path_wo_reorder = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp3/gradients"
    path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder/gradients"
    path_to_c_scores = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz"
    path_kl_2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp/gradients"
    path_mem = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feldman/mlp_cifar10_150_1000.pkl"
   
    tar_label = 0
    plot = "test"

    file_name = "output_compare"
    xcompare = "c"
    ycompare = "mem"
    zcompare = None
    x_label = xcompare
    y_label = ycompare
    show_images = False
    animate = True

    data_list = read_pkl_files_in_folder(path)
    data_list = limit_to_final(data_list)
    idx, kl1, kl2 = get_idx_kl1_kl2(data_list)
    kl_max = np.maximum(kl1, kl2)
    images, label = get_images_form_idx(idx)
    kl1_norm = [(kl1_id-min(kl_max))/ (max(kl_max) - min(kl_max)) for kl1_id in kl_max]
    [images, idx, kl_max, kl1, kl2] = limit_to_class([images, idx, kl_max, kl1, kl2], label, tar_label)

    
    if xcompare in ["kl12", "kl22"] or ycompare in ["kl12", "kl22"]or zcompare in ["kl12", "kl22"]:
        data_list2 = read_pkl_files_in_folder(path_kl_2)
        data_list2 = limit_to_final(data_list2)
        idx2, kl12, kl22 = get_idx_kl1_kl2(data_list2)
        common_indices, kl1, kl12 = find_common_indices_and_data(idx, idx2, kl1, kl12)
        common_indices, kl2, kl22 = find_common_indices_and_data(idx, idx2, kl2, kl22)

    if xcompare == "c" or ycompare == "c" or zcompare == "c":
        c_labels, c_scores = load_c_scores(path_to_c_scores)
        c_scores = c_scores[idx]
        c_labels = c_labels[idx]
        test = c_labels == label
        print(f"c_score correctness {all(test)}")
        [c_scores] = limit_to_class([c_scores], label, tar_label)


    print(f"Remaining idx: {len(idx)}")

    with open(path_mem, 'rb') as file:
        mem_data = pickle.load(file)
    mem_data = mem_data["data"]
    mem_list = []
    for index in idx:
        mem_list.append(mem_data.get(index, 0))
    [mem_list] = limit_to_class([mem_list], label, tar_label)



    if plot == "test":
        x_compare_data = None
        y_compare_data = None
        if xcompare == "mem":
            x_compare_data = mem_list
        elif xcompare == "kl1":
            x_compare_data = kl1
        elif xcompare == "kl2":
            x_compare_data = kl2
        elif xcompare == "kl12":
            x_compare_data = kl12
        elif xcompare == "kl22":
            x_compare_data = kl22
        elif xcompare == "c":
            x_compare_data = c_scores
        elif xcompare == "kl1_norm":
            x_compare_data = kl1_norm
        else:
            print("not found 1")

        if ycompare == "mem":
            y_compare_data = mem_list
        elif ycompare == "kl1":
            y_compare_data = kl1
        elif ycompare == "kl2":
            y_compare_data = kl2
        elif ycompare == "kl12":
            y_compare_data = kl12
        elif ycompare == "kl22":
            y_compare_data = kl22
        elif ycompare == "c":
            y_compare_data = c_scores
        elif ycompare == "kl1_norm":
            y_compare_data = kl1_norm
        else:
            print("not found 2")


        if zcompare == None:
            print("2D")
        elif zcompare == "mem":
            z_compare_data = mem_list
        elif zcompare == "kl1":
            z_compare_data = kl1
        elif zcompare == "kl2":
            z_compare_data = kl2
        elif zcompare == "kl12":
            z_compare_data = kl12
        elif zcompare == "kl22":
            z_compare_data = kl22
        elif zcompare == "c":
            z_compare_data = c_scores
        elif zcompare == "kl1_norm":
            z_compare_data = kl1_norm
        else:
            print("not found 2")

        if zcompare != None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            fig.set_dpi(500)
            fig.set_size_inches(10, 10)
            [x, y, z] = [x_compare_data, y_compare_data, z_compare_data]
            c1 = (x-np.min(x))/(np.max(x) - np.min(x))
            c2 = (y-np.min(y))/(np.max(y) - np.min(y))
            c3 =  (z-np.min(z))/(np.max(z) - np.min(z))
            c = np.array([c1, c2, c3]).transpose().tolist()
            plt.scatter(x_compare_data, y_compare_data, z_compare_data, depthshade=True, sizes =[30 for i in range(len(x_compare_data))], c=c)
            # ax.set_zlabel(zcompare)
            if animate:
                def animate(i):
                    ax.view_init(elev=10., azim=i)
                    return fig,
            
                def init():
                    ax.scatter(x_compare_data, y_compare_data, z_compare_data, depthshade=True, sizes =[30 for i in range(len(x_compare_data))], c=c)
                    return fig,

                # Animate
                anim = animation.FuncAnimation(fig, animate, init_func=init,
                                            frames=360, interval=20, blit=True)
                # Save
                anim.save('basic_animation.gif', fps=30)

        else:
            fig, ax = plt.subplots()
            fig.set_dpi(500)
            fig.set_size_inches(10, 10)
            plt.scatter(x_compare_data, y_compare_data)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.savefig(file_name + ".jpg")
        pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
    elif plot == "c_vs_KL":
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
        # imscatter(kl1, c_scores, images, zoom=0.5, ax=ax)
        plt.scatter(kl1, c_scores, c=color)
        ax.set_xlabel(r'$max(KL(D^{-1}||D), KL(D||D^{-1}))$')
        ax.set_ylabel(r'C-Score')
        plt.savefig("output_c_score.jpg")

main()