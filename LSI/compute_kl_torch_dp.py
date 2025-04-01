from Datasets.dataset_helper import get_dataset
import numpy as np
from laplace import Laplace
from utils.kl_div import _computeKL
import torch
from torch.utils.data import TensorDataset
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(512, n_classes)
        self.features = torch.nn.Sequential(self.linear1)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        return x

def get_mean_and_prec(data, labels, tinymodel, n_classes):
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels).to(torch.long)
    data = np.asarray(data)
    data = torch.from_numpy(data).to(torch.float32)

    model_intern = TinyModel(n_classes)
    with torch.no_grad():
        model_intern.linear1.weight = torch.nn.Parameter(tinymodel.linear1.weight.data)
        model_intern.linear1.bias = torch.nn.Parameter(tinymodel.linear1.bias.data)
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    la = Laplace(model_intern.features.to(DEVICE), 'classification',
                subset_of_weights='all',
                hessian_structure='diag')
    la.fit(train_loader)

    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_rem", type=int, default=5000, help="Value for lerining_rate (optional)")
    parser.add_argument("--batch_num", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--epochs", type=int, default=700, help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default= 0.01, help="Value for lerining_rate (optional)")
    parser.add_argument("--clip", type=float, default=0.001, help="Value for lerining_rate (optional)")
    parser.add_argument("--noise", type=float, default=0.0, help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=float, default=0.0)
    args = parser.parse_args()

    if args.dataset == "cifar100compressed":
        n_classes = 100
    if args.dataset == "cifar10compressed":
        n_classes = 10
    if args.dataset == "Primacompressed":
        n_classes = 3
        args.subset = 4646
        args.n_rem = 4646

    make_private = True

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_DP"
    if args.name == None:
        file_name = "kl_torch_epochs_" + str(args.epochs) + "_remove_" + str(args.n_rem) + "_dataset_" + str(args.dataset) + "_subset_" + str(args.subset) + "_corrupt_" + str(args.corrupt) + "_clip_" + str(args.clip) + "_noise_" + str(args.noise) + "_" + str(args.name_ext)
    else:
        file_name = args.name

    data_set_class, data_path = get_dataset(args.dataset)

    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    train_loader = DataLoader(data_set, batch_size=args.subset, shuffle=False)
    # len(data_set)
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)

    X_train = data_set.data.to(DEVICE)
    y_train = data_set.labels.to(DEVICE)
    X_test = data_set_test.data.to(DEVICE)
    y_test = data_set_test.labels.to(DEVICE)



    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    epochs = args.epochs
    i = 0
    kl = []
    idx = []
    pred = []
    square_diff = []
    acc = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for seed in range(N_SEEDS):
        seed_value = seed
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        privacy_engine = PrivacyEngine()
        model = TinyModel(n_classes)
        model = ModuleValidator.fix(model)
        backup_model = deepcopy(model)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.004,
            weight_decay=0.01, momentum=0.9, nesterov=True)
        if make_private:
            model, optimizer, train_loader_priv = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        noise_multiplier=args.noise,
                        max_grad_norm=args.clip,
                        noise_generator = generator,
                        loss_reduction = "mean"
                    )
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        mean1, prec1 = get_mean_and_prec(X_train.cpu(), y_train.cpu(), model, n_classes)
        kl_seed = []
        idx_seed = []
        pred_seed = []
        acc_seed = []
        for i in tqdm(range(N_REMOVE)):
            privacy_engine = PrivacyEngine()
            X_train_rem = torch.cat([X_train[0:i], X_train[i+1:]])
            y_train_rem = torch.cat([y_train[0:i], y_train[i+1:]])
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            model = deepcopy(backup_model)
            model = model.to(DEVICE)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.004,
                weight_decay=0.01, momentum=0.9, nesterov=True)
            if make_private:
                model, optimizer, _ = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        noise_multiplier=args.noise,
                        max_grad_norm=args.clip,
                        noise_generator = generator,
                        loss_reduction = "mean"
                    )
            for i in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_rem)
                pred = torch.argmax(output, dim=1)
                accuracy = torch.sum(pred == y_train_rem)/y_train_rem.shape[0]
                loss = criterion(output, y_train_rem)
                loss.backward()
                optimizer.step()
            print(accuracy.detach().item())
            acc_seed.append(accuracy)
            mean2, prec2 = get_mean_and_prec(X_train_rem.cpu(), y_train_rem.cpu(), model, n_classes)
            kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
            print(f"KL divergence {kl1}")
            kl_seed.append(kl1)
            square_diff.append(square_diff1)
            idx_seed.append(i)
        kl.append(kl_seed)
        idx.append(idx_seed)
        acc.append(acc_seed)

        
    
    result = {"idx": idx,
              "kl": kl,
              "pred":pred, 
              "square_diff":square_diff,
              "acc":acc}
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')