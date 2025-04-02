
from Datasets.dataset_helper import get_dataset
from laplace import Laplace
from copy import deepcopy
from LSI.experiments.utils_kl import _computeKL
from tqdm import tqdm
import os
import pickle
import time
import numpy as np
# from cuml.linear_model import LogisticRegression as cuLogisticRegression
# from cuml.metrics import accuracy_score
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from joblib import Parallel, delayed, effective_n_jobs, Memory
import multiprocessing


os.environ["CUPY_CACHE_DIR"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/cupy_cache_dir"
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(512, 10)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x
    
def get_mean_prec(data_set, max_iter, seed):
    torch.cuda.manual_seed(seed)
    X = data_set.data
    Y = data_set.labels
    X = X.numpy()
    Y = Y.numpy()

    start_time = time.time()
    clf = LogisticRegression(solver="lbfgs", random_state=seed, max_iter=max_iter, multi_class="multinomial", tol=0.01)
    clf.fit(X, Y)
    print(f"fit took {-start_time + time.time()}")
    train_score = clf.score(X, Y)
    # test_score = clf.score(data_set_test.data, data_set_test.labels)
    print(f"Train score: {train_score}")
    # print(f"Test score: {test_score}")

    # clf = cuLogisticRegression(solver="qn", max_iter=max_iter, tol=0.0001, fit_intercept=False)
    # start_time = time.time()
    # clf.fit(X, Y)
    # print(f"fit took {-start_time + time.time()} seconds")
    # train_predictions = clf.predict(X)
    # train_score = accuracy_score(Y, train_predictions)
    # print(f"Train score: {train_score}")
    # test_predictions = clf.predict(test_set.data.numpy())
    # test_score = accuracy_score(test_set.labels.numpy(), test_predictions)
    # print(f"Test score: {test_score}")
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    data_set.data = data_set.data.to(torch.float32)
    data_set.labels = data_set.labels.to(torch.long)
    tinymodel = TinyModel()
    weights = torch.nn.Parameter(torch.Tensor(clf.coef_)).to(torch.float32)
    bias = torch.nn.Parameter(torch.Tensor(clf.intercept_)).to(torch.float32)
    with torch.no_grad():
        tinymodel.linear1.weight = weights
        tinymodel.linear1.bias = bias

    # for _, (data, target, idx, _) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     output = tinymodel(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    la = Laplace(tinymodel.features, 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec    

def compute_loo_kl(input):
    data_set_loc, indices = input
    result = []
    for i in indices:
        data_set_rm = deepcopy(data_set_loc)
        data_set_rm.remove_curr_index_from_data(i)
        mean2, prec2 = get_mean_prec(data_set_rm, N_MAX_ITER, 0)
        kl1 = _computeKL(mean1, mean2, prec1, prec2)
        kl2 = _computeKL(mean2, mean1, prec2, prec1)
        print(f"----- KL1 {kl1}, KL2 {kl2}")
    result.append([kl1, kl2, i])
    return result

def create_nested_list(values, c):
    result_list = [[n, c] for n in values]
    return result_list


def get_cpu_count():
    return effective_n_jobs()

def split_range_evenly(n, m):
    # Generate indices for splitting the range
    indices = np.linspace(0, m, num=n + 1, dtype=int)

    # Create lists based on the generated indices
    result = [list(range(indices[i], indices[i + 1])) for i in range(n)]

    return result

# Get the number of CPUs
cpu_count = get_cpu_count()
print(f"CPU count {cpu_count}")


data_set_class, data_path = get_dataset("cifar10compressed")

data_set = data_set_class(data_path, train=True)
# keep_indices = [*range(7500)]
# data_set.reduce_to_active(keep_indices)
data_set_test = data_set_class(data_path, train=False) 
data_set.data = data_set.data.to(torch.float32)
data_set.labels = data_set.labels.to(torch.float32)
mem = Memory(location='.', mmap_mode='r')

N_REMOVE = 50
N_MAX_ITER = 10000
mean1, prec1 = get_mean_prec(data_set, N_MAX_ITER, 0)
# for i in range(N_REMOVE):
#     compute_loo_kl((data_set_test, data_set, i, j))
#     result_dict[j][i] = kl1
# items_to_process = list(range(N_REMOVE))
num_jobs = cpu_count # min(len(items_to_process), cpu_count)
items_to_process = split_range_evenly(num_jobs, N_REMOVE)
results = Parallel(n_jobs=num_jobs)(delayed(compute_loo_kl)([deepcopy(data_set), item]) for item in items_to_process)
print("")
final_save_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_scikit"
if not os.path.exists(final_save_path):
    os.makedirs(final_save_path)
with open(final_save_path + "/results_all" + str(N_REMOVE) + "_" + str(N_MAX_ITER) + ".pkl", 'wb') as file:
    pickle.dump(results, file)
print(f'Saving at {final_save_path + "/results_all" + str(N_REMOVE) + "_" + str(N_MAX_ITER) + ".pkl"}')