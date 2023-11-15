from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from Datasets.dataset_helper import get_dataset
import sk2torch
import torch
from laplace import Laplace
from copy import deepcopy
from utils.kl_div import _computeKL
from tqdm import tqdm
import os
import pickle


def get_mean_prec(data_set, seed):
    X = data_set.data
    Y = data_set.labels
    clf = LogisticRegression(solver="lbfgs", random_state=seed, max_iter=5000, multi_class="multinomial")
    clf.fit(X, Y)
    train = clf.score(X, Y)
    print(train)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    class TinyModel(torch.nn.Module):
        def __init__(self):
            super(TinyModel, self).__init__()
            self.linear1 = torch.nn.Linear(512, 100)
            self.features = torch.nn.Sequential(self.linear1)


        def forward(self, x):
            x = x.to(torch.float32)
            x = self.features(x)
            return x

    data_set.data = data_set.data.to(torch.float32)
    data_set.labels = data_set.labels.astype(int)
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

data_set_class, data_path = get_dataset("cifar100compressed")

data_set = data_set_class(data_path, train=True)
data_set_test = data_set_class(data_path, train=False) 
data_set.data = data_set.data.to(torch.float64)
data_set.labels = data_set.labels.astype(float)

result_dict = {}
N_SEEDS = 1
N_REMOVE = 100
for j in tqdm(range(N_SEEDS)):
    result_dict[j] = {}
    mean1, prec1 = get_mean_prec(data_set, j)
    for i in range(N_REMOVE):
        data_set_rm = deepcopy(data_set)
        data_set_rm.remove_curr_index_from_data(i)
        mean2, prec2 = get_mean_prec(data_set_rm, j)
        kl1 = _computeKL(mean1, mean2, prec1, prec2)
        result_dict[j][i] = kl1
print("")
final_save_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_scikit"
if not os.path.exists(final_save_path):
    os.makedirs(final_save_path)
with open(final_save_path + "/results_all.pkl", 'wb') as file:
    pickle.dump(result_dict, file)
print(f'Saving at {final_save_path + "/results_all.pkl"}')