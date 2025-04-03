import os
import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")
import io
import pickle
import torch
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



def get_color(images):
    colors = []
    for image in images:
        colors.append(np.mean(image))
    return colors

def get_kl_data(final_path, agg_type):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict[agg_type]
    idx = list(kl_data.keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data = [kl_data[str(index)] for index in idx]
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
        modified_values = values[:removed_index] + [None] + values[removed_index:]
        
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
    means_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace_cifar_long500_idx_0"

    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_final_kl_all/final.pkl"

    kl_data, idx = get_kl_data(kl_path, "kl2_max")

    # kl_data, idx = get_kl_data(final_path)
    averages = get_rdp(means_path, idx)
    images, label = get_images_form_idx(idx)



    tar_label = None
    plot = "test"

    file_name = "z_kl_vs_grad_norm"
    xcompare = averages
    ycompare = kl_data
    zcompare = None
    x_label = "sum_grad_norm"
    y_label = "kl_2_max"
    show_images = False
    animate = True

    x_compare_data = xcompare
    y_compare_data = ycompare


    if plot == "test":     
        # x_compare_data_temp = [] 
        # y_compare_data_temp = []  
        # for (x_com, y_com) in zip(x_compare_data, y_compare_data):
        #     if x_com > 0.0 and x_com < 2:
        #         x_compare_data_temp.append(x_com)
        #         y_compare_data_temp.append(y_com)
        # x_compare_data = x_compare_data_temp
        # y_compare_data = y_compare_data_temp
                

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
            if show_images:
                imscatter(x_compare_data[0:50], y_compare_data[0:50], images[0:50])
            else:
                plt.scatter(x_compare_data, y_compare_data)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.savefig(file_name + ".jpg")
        print(f"saving fig as {file_name}.jpg")

main()