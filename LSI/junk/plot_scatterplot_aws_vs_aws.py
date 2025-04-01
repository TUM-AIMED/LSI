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
    dataset_class, data_path = get_dataset("mnist")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    data_set._set_classes([4, 9]) # mnist
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

def get_aws_data(path, index=None):
    with open(path, 'rb') as file:
        final_dict = CPU_Unpickler(file).load()
    data = final_dict["importance_measures"]
    if index:
        data = data[index]
    data = [elem.item() for elem in data]
    return data



def main():
    lin_path = "/vol/aimspace/users/kaiserj/aws-cv-unique-information/results/newer/weight-plain-t200-h854439/results.pkl"
    lap_path = "/vol/aimspace/users/kaiserj/aws-like-kl-divergence/results/mnist_4_v_9_log_weigh_norm/predictions-t200-h825227/results.pkl"
    # lin_path2 = "/vol/aimspace/users/kaiserj/aws-cv-unique-information/results/test/predictions-t200-h337823/results.pkl"

    idx = [*range(1000)]
    lin_data = get_aws_data(lin_path)
    lap_data = get_aws_data(lap_path, 2)
    
    
    images, label = get_images_form_idx(idx)



    tar_label = None
    plot = "test"

    file_name = "z_lin_aws_vs_lap_aws"
    xcompare = lin_data
    ycompare = lap_data
    zcompare = None
    x_label = "lin_data"
    y_label = "lap_data"
    show_images = True
    animate = True

    x_compare_data = xcompare
    y_compare_data = ycompare


    fig, ax = plt.subplots()
    fig.set_dpi(2000)
    fig.set_size_inches(10, 10)
    if show_images:
        imscatter(x_compare_data[0:200], y_compare_data[0:200], images[0:200])
    else:
        plt.scatter(x_compare_data, y_compare_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as {file_name}.jpg")

main()