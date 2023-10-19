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


def load_c_scores(path):
    c_data = np.load(path)
    labels = c_data["labels"]
    scores = c_data["scores"]
    return labels, scores


def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_final_kl_all/final.pkl"
    c_score_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz"


    kl_data_max, idx = get_kl_data(kl_path, "kl2_max")
    images, label = get_images_form_idx(idx)

    clabels, cscores = load_c_scores(c_score_path)
    cscores = [cscores[index] for index in idx]
    clabels = [clabels[index] for index in idx]

    test = clabels == label

    tar_label = None
    plot = "test"

    file_name = "z_c_vs_kl"
    xcompare = kl_data_max
    ycompare = cscores
    zcompare = None
    x_label = "kl_data_max"
    y_label = "c-scores"
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