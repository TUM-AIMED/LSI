import os
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
                    # print(f"Loaded data from '{filename}'")
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
    dataset_class, data_path = get_dataset("mnist")
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
    if tar_label == None:
        return data
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


def get_color(images):
    colors = []
    for image in images:
        colors.append(np.mean(image))
    return colors


def main():
    path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace_cifar_long500_idx_0"
    path_compare = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist4/compare"
    path_to_c_scores = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz"
    path_kl_2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_mlp/gradients"
    path_mem = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld/feldman_cifar10_400.pkl"
    path_priv = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp2/gradients/idp_try_2173.pkl"
    path_priv2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp2/gradients/idp_try_2171.pkl"
    path_inform = "/vol/aimspace/users/kaiserj/aws-cv-unique-information/results/test/predictions-t2000-h600849/results.pkl"
    path_kl_replace ="/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace/final/final.pkl"
    path_kl_non_replace = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist6/gradients"
    path_means = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_replace_cifar_long500_idx_0"

    tar_label = None
    plot = "test"

    file_name = "z_kl_vs_priv4"
    xcompare = "kl1_max"
    ycompare = "kl1_non_replace"
    zcompare = None
    x_label = xcompare
    y_label = ycompare
    show_images = True
    animate = True


    accuracies = read_pkl_files_in_folder(path_compare)[0][0]["per_class_accuracies"]
    for k, v in accuracies.items():
        print(f"{k}: {v}")


    with open(path_kl_replace, 'rb') as file:
        data_kl = pickle.load(file)
    kl1_mean = data_kl["kl1_mean"]
    kl2_mean = data_kl["kl2_mean"]
    kl1_max = data_kl["kl1_max"]
    kl2_max = data_kl["kl2_max"]

    idx = list(kl1_mean.keys())
    values = list(kl1_mean.values())
    idx.sort()
    result_list = [kl1_mean[index] for index in idx]
    kl1 = result_list
    result_list = [kl2_mean[index] for index in idx]
    kl2 = result_list
    
    result_list = [kl1_max[index] for index in idx]
    kl1_max = result_list
    result_list = [kl2_max[index] for index in idx]
    kl2_max = result_list
    


    data_list = read_pkl_files_in_folder(path_kl_non_replace)
    data_list = limit_to_final(data_list)
    idx_non_replace, kl1_non_replace, kl2_non_replace = get_idx_kl1_kl2(data_list)
    images, label = get_images_form_idx(idx)
    # [images, idx, kl1, kl2] = limit_to_class([images, idx, kl1, kl2], label, tar_label)
    kl1_non_replace = [kl1_non_replace[index] for index in idx]
    kl2_non_replace = [kl2_non_replace[index] for index in idx]

    color_data = get_color(images)


    # if xcompare in ["kl1_norm", "kl2_norm", "kl_norm_max", "kl_norm_min"] or ycompare in ["kl1_norm", "kl2_norm", "kl_norm_max", "kl_norm_min"] or zcompare in ["kl1_norm", "kl2_norm", "kl_norm_max", "kl_norm_min"]:
    #     kl1_norm = [(kl1_id-min(kl1))/ (max(kl1) - min(kl1)) for kl1_id in kl1]
    #     kl2_norm = [(kl2_id-min(kl2))/ (max(kl2) - min(kl2)) for kl2_id in kl2]
    #     kl_norm_max = [max(x, y) for x, y in zip(kl1_norm, kl2_norm)]
    #     kl_norm_min = [min(x, y) for x, y in zip(kl1_norm, kl2_norm)]
    #     [kl1_norm, kl2_norm, kl_norm_max, kl_norm_min] = limit_to_class([kl1_norm, kl2_norm, kl_norm_max, kl_norm_min], label, tar_label)
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

    if xcompare == "priv" or ycompare == "priv" or zcompare == "priv":
        with open(path_priv, 'rb') as file:
            priv_data_init = pickle.load(file)
            priv_data_init = priv_data_init[-1]
            priv_data = priv_data_init["idp_accountant"]
            labels_priv = priv_data_init["labels"]
            print("")
            priv_data = [priv_data[index] for index in idx]
            labels_priv_orderd = [labels_priv[index].item() for index in idx]
            test = labels_priv_orderd == label
            if test:
                print("Sanity-check iDP successfull")
            else:
                raise Exception("Sanity-check idp failed")
    if xcompare == "priv2" or ycompare == "priv2" or zcompare == "priv2":
        with open(path_priv2, 'rb') as file:
            priv_data_init2 = pickle.load(file)
            priv_data_init2 = priv_data_init2[-1]
            priv_data2 = priv_data_init2["idp_accountant"]
            labels_priv2 = priv_data_init2["labels"]
            print("")
            priv_data2 = [priv_data2[index] for index in idx]
            labels_priv_orderd2 = [labels_priv2[index].item() for index in idx]
            test = labels_priv_orderd2 == label
            if test:
                print("Sanity-check iDP successfull")
            else:
                raise Exception("Sanity-check idp failed")

    if xcompare == "inform" or ycompare == "inform" or zcompare == "inform":
        with open(path_inform, 'rb') as file:
            data_inform = CPU_Unpickler(file).load()
        data_imp_measures = data_inform["importance_measures"]
        data_imp_measures = [data_imp.item() for data_imp in data_imp_measures]
        labels_inform = data_inform["labels"]
        labels_inform = [torch.argmax(tensor).item() for tensor in labels_inform]
    
        data_inform_ordered = [data_imp_measures[index] for index in idx]
        labels_inform_orderd = [labels_inform[index] for index in idx]
        test = labels_inform_orderd == label
        if test:
            print("Sanity-check inform successfull")
        else:
            raise Exception("Sanity-check inform failed")
        

    print(f"Remaining idx: {len(idx)}")
    if xcompare == "mem" or ycompare == "mem" or zcompare == "mem":
        with open(path_mem, 'rb') as file:
            mem_data = pickle.load(file)
        mem_data = mem_data["data"]
        mem_list = []
        for index in idx:
            mem_list.append(mem_data.get(index, 0))
        [mem_list] = limit_to_class([mem_list], label, tar_label)

    def get_data_from_name(compare):
        compare_data = None
        if compare == None:
            compare_data == None
        elif compare == "mem":
            compare_data = mem_list
        elif compare == "kl1":
            compare_data = kl1
        elif compare == "kl2":
            compare_data = kl2
        elif compare == "kl1_max":
            compare_data = kl1_max
        elif compare == "kl2_max":
            compare_data = kl2_max
        elif compare == "kl1_non_replace":
            compare_data = kl1_non_replace
        elif compare == "kl12":
            compare_data = kl12
        elif compare == "kl22":
            compare_data = kl22
        elif compare == "c":
            compare_data = c_scores
        elif compare == "priv":
            compare_data = priv_data
        elif compare == "priv2":
            compare_data = priv_data2
        elif compare == "inform":
            compare_data = data_inform_ordered
        elif compare == "kl1_norm":
            compare_data = kl1_norm
        elif compare == "kl2_norm":
            compare_data = kl2_norm
        elif compare == "kl_norm_max":
            compare_data = kl_norm_max
        elif compare == "kl_norm_min":
            compare_data = kl_norm_min
        elif compare == "color":
            compare_data = color_data
        else:
            raise Exception(f"{compare} not found")
        return compare_data

    if plot == "test":     
        x_compare_data = get_data_from_name(xcompare)
        y_compare_data = get_data_from_name(ycompare)
        z_compare_data = get_data_from_name(zcompare)    

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
                imscatter(x_compare_data, y_compare_data, images)
            else:
                plt.scatter(x_compare_data, y_compare_data)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.savefig(file_name + ".jpg")
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