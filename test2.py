import os
import pickle

folder_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder/compare/compare_0.pkl"


# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pkl"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                print(f"Loaded data from '{filename}'")
        except Exception as e:
            print(f"Error loading data from '{filename}': {str(e)}")