import pickle
import matplotlib.pyplot as plt
def load_data_from_pickle(path1, path2):
    try:
        with open(path1, 'rb') as file1:
            data1 = pickle.load(file1)
        with open(path2, 'rb') as file2:
            data2 = pickle.load(file2)
        return data1, data2
    except FileNotFoundError:
        print("One or both of the files could not be found.")
        return None, None
    except Exception as e:
        print("An error occurred while loading data:", e)
        return None, None
def create_scatter_plot_and_save(list1, list2, filename):
    # Check if both lists have the same length
    if len(list1) != len(list2):
        print("Error: Both lists must have the same length.")
        return

    # Create scatter plot
    plt.scatter(list1, list2)
    plt.xlabel('List 1')
    plt.ylabel('List 2')
    plt.title('Scatter Plot')

    # Save plot as PNG
    plt.savefig(filename)
    plt.close()

# Example usage:
path_to_pickle1 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/diag.pkl"
path_to_pickle2 = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/full.pkl"
data1, data2 = load_data_from_pickle(path_to_pickle1, path_to_pickle2)
data1 = data2[0][0:10]
data2 = data2[1][0:10]
output_filename = "./full_vs_diag.png"
create_scatter_plot_and_save(data1, data2, output_filename)
print("")