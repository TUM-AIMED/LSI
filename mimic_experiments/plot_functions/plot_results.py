import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def load_data(data_path):
    with open(data_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def plot_bar(values, colormap, fig_num):
    color_mapping = {
        1: 'red',
        2: 'green',
        0: 'blue',
        4: 'orange',
        5: 'purple'
    }
    colors = [color_mapping[param] for param in colormap]
    f = plt.figure(fig_num)
    x_ticks = range(len(values))
    cmap = plt.get_cmap('Set3')
    plt.bar(x_ticks, values, color=colors)
    # plt.xticks(x_ticks)
    f.show()


def plot_scatter(values_1, values_2, colormap, fig_num):
    color_mapping = {
        1: 'red',
        2: 'green',
        0: 'blue',
        4: 'orange',
        5: 'purple'
    }
    colors = [color_mapping[param] for param in colormap]
    f = plt.figure(fig_num)
    plt.scatter(values_1, values_2, color=colors, s=10)
    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')

    # Show the plot
    f.show()
    
def prep_data(data, selection):
    highes_idx = max(data[0].idx)
    summed_data = np.zeros(highes_idx+1)
    for iteration, data_per_iter in enumerate(data):
        summed_data[data_per_iter.idx] += data_per_iter[selection]
    summed_data = [summed_data[i] for i in data[0].idx]

    colormap = np.zeros(highes_idx+1)
    colormap[data[0].idx] += data[0].classes
    colormap = [colormap[i] for i in data[0].idx]
    return summed_data, colormap

if __name__ == "__main__":
    data_path = "./results/gradients/trial.pkl"
    gradients_active = []
    gradients_complete = []
    losses_active = []
    losses_complete = []
    data = load_data(data_path)
    for iteration in data:
        gradients_active.append(iteration["gradient_df_active"])
        gradients_complete.append(iteration["gradient_df_complete"])
        losses_active.append(iteration["losses_df_active"])
        losses_complete.append(iteration["losses_df_complete"])

    summed_grad_c, colormap = prep_data(gradients_complete, "total_gradient_norm")
    plot_bar(summed_grad_c, colormap, 1)

    summed_grad_a, colormap = prep_data(gradients_active, "total_gradient_norm")
    plot_bar(summed_grad_a, colormap, 2)

    summed_loss_a, colormap = prep_data(losses_active, "losses")
    # plot_bar(summed_data, colormap, 3)

    summed_loss_c, colormap = prep_data(losses_complete, "losses")
    # plot_bar(summed_data, colormap, 4)

    # plot_scatter(summed_loss_c, summed_grad_c, colormap, 1)
    input()
