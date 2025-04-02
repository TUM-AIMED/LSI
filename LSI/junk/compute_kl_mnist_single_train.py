import sys
import os
import pickle

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from opacus.validators.module_validator import ModuleValidator
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import warnings
import argparse
import time
import random
from laplace.curvature import AsdlGGN, BackPackGGN, AsdlHessian, AsdlEF, BackPackEF
from LSI.experiments.utils_kl import computeKL
from utils.plot_utils import plot_and_save_histogram, plot_and_save_lineplot, plot_and_save_lineplot_with_running_sum


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_active):
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    start_time = time.time()
    total = 0
    correct = 0
    loss_list = []
    for _, (data, target, idx, _) in enumerate(train_loader_active):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    # print(f"training accuracy {accuracy:.4f}, epoch took {time.time() - start_time:.4f}")
    return np.mean(np.array(loss_list)), accuracy


def normal_val_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               val_loader_active):
    start_time = time.time()
    total = 0
    correct = 0
    loss_list = []
    for _, (data, target, idx, _) in enumerate(val_loader_active):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss_list.append(loss.item())
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return np.mean(np.array(loss_list)), accuracy


def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion,
    scheduler = None,
    indiv_Accountant = None,
    rem = False
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    print(f'File name: {params["model"]["name"]}')
    print(f'lr: {params["training"]["learning_rate"]}', flush=True)
    print(f'l2: {params["training"]["l2_regularizer"]}', flush=True)
    print(f'epochs: {params["training"]["num_epochs"]}', flush=True)
    print(f'batchsize: {params["training"]["batch_size"]}', flush=True)

    early_stopper = EarlyStopper(patience=25, min_delta=0.2)    

    for epoch in range(params["training"]["num_epochs"]):
        model.train()
        train_loss, test_accuracy = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0,)
        model.eval()
        if epoch % params["testing"]["test_every"] == 0:
            val_loss, val_accuracy = normal_val_step(params,
                model, 
                optimizer, 
                DEVICE, 
                criterion, 
                test_loader)
        if not rem:
            if early_stopper.early_stop(val_loss): 
                print(f"Stopping early at epoch {epoch}")            
                break
        if scheduler != None:
            scheduler.step()
        # print(f"Epoch {epoch} with training_loss {train_loss} and val_loss {val_loss}")
        print(f"train accuracy {test_accuracy:.4f}")
    print(f"val accuracy {val_accuracy:.4f}")
    return model, epoch


def train_with_params(
    params,
    train_loader_0,
    test_loader,
    rem = False
):
    """
    train_with_params initializes the main training parts (model, criterion, optimizer and makes private)

    :param params: dict with all parameters
    """ 

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params["model"]["seed"])
    torch.cuda.manual_seed_all(params["model"]["seed"])
    np.random.seed(params["model"]["seed"])
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

     
    

    train_X_example, _, _, _ = train_loader_0.dataset[0]
    N = len(train_loader_0.dataset)
    print(f"Length = {N}")

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    # if N < params["training"]["batch_size"]:
    #     raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = model_class(len(train_X_example), N_CLASSES)
        model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"], momentum=0.9, nesterov=True)
    # optimizer = torch.optim.LBFGS(
    #     model.parameters(),
    #     max_iter=20,
    #     lr=1e-3,
    #     line_search_fn="strong_wolfe") 
    scheduler = None # torch.optim.lr_scheduler.MultiStepLR(optimizer, [700, 1000, 1200, 1400], gamma=0.5)


    return criterion, train(
                        params,
                        model,
                        DEVICE,
                        train_loader_0,
                        test_loader,
                        optimizer,
                        criterion,
                        scheduler = scheduler,
                        rem=rem
                    )




if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_rem", type=int, default=400, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default="cifar_cnn_100_400_once", help="Value for lerining_rate (optional)")
    parser.add_argument("--repr", type=str, nargs='*', default=["diag"], help="Value for lerining_rate (optional)")
    parser.add_argument("--lap_type", type=str, default="asdlgnn", help="Value for lerining_rate (optional)")
    args = parser.parse_args()

    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    try_num = 0
    if N_REMOVE == 0:
        N_REMOVE = len(data_set.labels)

    if args.lap_type == "asdlgnn":
        backend_class = AsdlGGN
    elif args.lap_type == "asdlhessian":
        backend_class = AsdlHessian
    elif args.lap_type == "asdlef":
        backend_class = AsdlEF
    elif args.lap_type == "backpackgnn":
        backend_class = BackPackGGN
    elif args.lap_type == "backpackef":
        backend_class = BackPackEF
    else:
        raise Exception("Not implemented")
    
    representation = args.repr

    params = {}
    params["save"] = True
    params["model"] = {}
    params["model"]["seed"] = 472168
    params["model"]["model"] = "logreg4"
    params["model"]["dataset_name"] = "cifar100compressed"
    params["model"]["name_base"] = "laplace_mnist_"
    params["model"]["name"] = "laplace_mnist_"
    params["model"]["name"] = params["model"]["name_base"] + str(try_num)
    params["training"] = {}
    params["training"]["batch_size"] = 256
    params["training"]["learning_rate"] = 1e-02# 0.002 mlp #3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-08
    params["training"]["num_epochs_init"] = 1000
    params["testing"] = {}
    params["testing"]["test_every"] = params["training"]["num_epochs_init"] + 1
    params["Paths"] = {}
    params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_" + str(args.name)

    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])

    data_set = data_set_class(data_path, train=True)

    keep_indices = [*range(50000)]
    data_set.reduce_to_active(keep_indices)
    # random_labels_idx = None
    # n = int(0.05 * len(keep_indices))
    # random_labels_idx = random.sample([*range(N_REMOVE)], n)
    # data_set.apply_label_noise(random_labels_idx)
    # # # data_set.apply_image_mark(random_labels_idx)
    # # random_labels_idx = data_set.apply_group_label_noise(under=N_REMOVE)


    data_set_test = data_set_class(data_path, train=False)
    # data_set_test._set_classes([0, 1])
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )



    resultskl1_diag = {}
    resultskl2_diag = {}
    resultskl1_kron = {}
    resultskl2_kron = {}
    resultskl1_full = {}
    resultskl2_full = {}
    resultstarget = {}
    mean_diff_sum = {}
    mean_diff_mean = {}
    for seed in tqdm(range(N_SEEDS)):
        params["training"]["num_epochs"] = params["training"]["num_epochs_init"]
        resultskl1_diag[seed] = {}
        resultskl2_diag[seed] = {}
        resultskl1_kron[seed] = {}
        resultskl2_kron[seed] = {}
        resultskl1_full[seed] = {}
        resultskl2_full[seed] = {}
        resultstarget[seed] = {}
        mean_diff_sum[seed] = {}
        mean_diff_mean[seed] = {}

        params["model"]["seed"] = seed

        train_loader_0 = torch.utils.data.DataLoader(
                data_set,
                batch_size=len(data_set), # params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
        criterion, (model_all, epochs) = train_with_params(
            params,
            train_loader_0,
            test_loader
        )
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
        for rem_idx in range(N_REMOVE):
            print("--------------------------")
            print(f"Compute Remove Model {rem_idx}", flush=True)
            print("--------------------------")
            data_set_rm = deepcopy(data_set)
            data_set_rm.remove_curr_index_from_data(rem_idx)
            
            train_loader_rm = torch.utils.data.DataLoader(
                data_set_rm,
                batch_size=len(data_set_rm), # params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
            true_rm_idx = data_set.active_indices[rem_idx]
            # true_rm_idx2 = keep_indices[rem_idx]
            # if true_rm_idx != true_rm_idx2:
            #     raise Exception("Something went wrong here")
            
            start_time = time.time()
            if "kron" in representation:
                print("Compute Kron", flush=True)
                kl1_1, kl2_1, mean_diff_s, mean_diff_m, _, _ = computeKL(DEVICE, backend_class, "kron", model_all, model_all, train_loader_0, train_loader_rm, params["training"]["l2_regularizer"]) 
                resultskl1_kron[seed][true_rm_idx] = kl1_1
                resultskl2_kron[seed][true_rm_idx] = kl2_1
            if "diag" in representation:
                print("Compute Diag", flush=True)
                kl1_2, kl2_2, mean_diff_s, mean_diff_m, _, _ = computeKL(DEVICE, backend_class, "diag", model_all, model_all, train_loader_0, train_loader_rm, params["training"]["l2_regularizer"]) 
                resultskl1_diag[seed][true_rm_idx] = kl1_2
                resultskl2_diag[seed][true_rm_idx] = kl2_2
            if "full" in representation:
                print("Compute Full", flush=True)
                kl1_3, kl2_3, mean_diff_s, mean_diff_m, _, _ = computeKL(DEVICE, backend_class, "full", model_all, model_all, train_loader_0, train_loader_rm, params["training"]["l2_regularizer"]) 
                resultskl1_full[seed][true_rm_idx] = kl1_3
                resultskl2_full[seed][true_rm_idx] = kl2_3
            print(f"kl computation took {time.time() - start_time}", flush=True)            

            mean_diff_sum[seed][true_rm_idx] = mean_diff_s
            mean_diff_mean[seed][true_rm_idx] = mean_diff_m
            resultstarget[seed][true_rm_idx] = data_set.labels[rem_idx]

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    results_all = {
        "kl1_diag" : resultskl1_diag,
        "kl2_diag" : resultskl2_diag,
        "kl1_kron" : resultskl1_kron,
        "kl2_kron" : resultskl2_kron,
        "kl1_full" : resultskl1_full,
        "kl2_full" : resultskl2_full,
        "labels": resultstarget,
        "mean_diff_sum": mean_diff_sum,
        "mean_diff_mean": mean_diff_mean,
        "random_labels_idx": None
    }
    with open(params["Paths"]["final_save_path"] + "/results_single_layer_train.pkl", 'wb') as file:
        pickle.dump(results_all, file)
    print(f'Saving at {params["Paths"]["final_save_path"] + "/results_single_layer_train.pkl"}')

