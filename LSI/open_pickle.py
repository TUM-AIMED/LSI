import pickle

final_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_scikit/results_all.pkl"

with open(final_path, 'rb') as file:
    final_dict = pickle.load(file)
print("")