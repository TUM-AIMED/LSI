import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_histogram(values, filename, bins="auto"):
    # Create the histogram
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, color='blue', alpha=0.7)
    
    # Adding title and labels
    ax.set_title('Histogram of Given Values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Save the plot as an image file
    plt.savefig(filename)
    


def plot_and_save_lineplot(x, y, filename):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(120, 4))
    
    # Create the line plot on the specified axis
    ax.plot(x, y, color='blue')
    
    # Adding title and labels
    ax.set_title('Line Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    # Save the plot as an image file
    plt.savefig(filename)


def plot_and_save_lineplot_with_running_sum(x, y, filename):
    # Create a figure
    fig, ax1 = plt.subplots(figsize=(120, 4))

    # Create the line plot on the primary axis
    ax1.plot(x, y, color='blue', label='Line Plot')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis', color='blue')
    
    # Calculate and plot the running sum on a secondary axis
    running_sum = np.cumsum(y)
    ax2 = ax1.twinx()  # Create a secondary axis that shares the same x-axis
    ax2.plot(x, running_sum, color='red', label='Running Sum')
    ax2.set_ylabel('Running Sum', color='red')

    # Adding title and legend
    plt.title('Line Plot with Running Sum')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    plt.legend(lines, labels)

    # Save the plot as an image file
    plt.savefig(filename)