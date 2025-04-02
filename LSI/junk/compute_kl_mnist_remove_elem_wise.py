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


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


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
               train_loader_active,
               data=None,
               target=None,
               idx=None):
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    t_time = time.time()
    total = 0
    correct = 0
    accuracy = 0
    loss_list = []
    if data == None:
        if isinstance(optimizer, torch.optim.LBFGS):
            for _, (data, target, idx, _) in enumerate(train_loader_active):
                data, target = data.to(DEVICE), target.to(DEVICE)
                def closure():
                    optimizer.zero_grad()
                    output = model(data)
                    l2_norm = torch.tensor(0.).cpu()
                    for p in model.parameters():
                        l2_norm += (p**2).sum().cpu()
                    loss = criterion(output, target)
                    loss += params["training"]["l2_regularizer"] * l2_norm
                    loss.backward()
                    return loss
                optimizer.step(closure)
                output = model(data)
                loss = closure()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss_list.append(loss.item())
                accuracy = correct/total
            torch.cuda.empty_cache()
            # print(f"training accuracy {accuracy:.4f}, loss {loss:.4f} epoch took {time.time() - start_time:.4f}")
            return np.mean(np.array(loss_list)), accuracy
        else:
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
            # print(f"training accuracy {accuracy:.4f}, loss {loss:.4f}, epoch took {time.time() - start_time:.4f}")
            return np.mean(np.array(loss_list)), accuracy
    else:
        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                l2_norm = torch.tensor(0.).cpu()
                for p in model.parameters():
                    l2_norm += (p**2).sum().cpu()
                loss = criterion(output, target)
                loss += params["training"]["l2_regularizer"] * l2_norm
                loss.backward()
                return loss
            optimizer.step(closure)
            output = model(data)
            loss = closure()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss_list.append(loss.item())
            accuracy = correct/total
            torch.cuda.empty_cache()
            # print(f"training accuracy {accuracy:.4f}, loss {loss:.4f} epoch took {time.time() - start_time:.4f}")
            return np.mean(np.array(loss_list)), accuracy
        else:
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
            # print(f"training accuracy {accuracy:.4f}, loss {loss:.4f}, epoch took {time.time() - start_time:.4f}")
            return np.mean(np.array(loss_list)), accuracy
        
def normal_val_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               val_loader_active,):
    start_time = time.time()
    total = 0
    correct = 0
    accuracy = 0
    loss_list = []
    for _, (data, target, idx, _) in enumerate(val_loader_active):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total += target.size(0)
        _, predicted = torch.max(output.data, 1)
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
    train_loader_rm = None,
    rem = False
):

    early_stopper = EarlyStopper(patience=5000, min_delta=0.2) 
    data = None
    target = None
    idx = None   
    if params["Full_Batch"]:
        data, target, idx, _ = next(iter(train_loader_0))
        data, target = data.to(DEVICE), target.to(DEVICE)
    for epoch in range(params["training"]["num_epochs"]):
        # print(epoch)
        model.train()
        train_loss, train_accuracy = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0,
               data=data,
               target=target,
               idx=idx)
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
    return model, train_accuracy


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
    torch.use_deterministic_algorithms(True)
     
    

    train_X_example, _, _, _ = train_loader_0.dataset[0]
    N = len(train_loader_0.dataset)

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = model_class(len(train_X_example), N_CLASSES)
        model = model.to(DEVICE)
    if params["freeze"] == True:
        model.freeze_all_but_last() 
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
    parser.add_argument("--n_seeds", type=int, default=1, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--repr", type=str, nargs='*', default=["diag"], help="Value for lerining_rate (optional)")
    parser.add_argument("--lap_type", type=str, default="asdlgnn", help="Value for lerining_rate (optional)")
    parser.add_argument("--freeze", type=bool, default=False, help="Value for lerining_rate (optional)")
    parser.add_argument("--epochs", type=int, default=25, help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="logreg4", help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar100compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--kllayer", type=str, default="all", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=3e-1, help="Value for lerining_rate (optional)")

    args = parser.parse_args()
    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    N_SUBSET_ORDERING = args.n_seeds
    try_num = 0

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
    params["model"]["model"] = args.model
    params["model"]["dataset_name"] = args.dataset
    params["training"] = {}
    params["training"]["batch_size"] = 50
    params["training"]["learning_rate"] = args.lr # 1e-2 #3e-1 for SGD on cifar100 # 6e-03 #cifar100# 2e-3# cifar10 logreg2/ cifar100 logreg3 1k/10k# 4e-02# 0.002 mlp #3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-8
    params["training"]["num_epochs_init"] = args.epochs
    params["testing"] = {}
    params["testing"]["test_every"] = 1 # params["training"]["num_epochs_init"] + 1
    params["Paths"] = {}
    params["Full_Batch"] = True

    params["freeze"] = args.freeze
    params["kllayer"] = args.kllayer

    
    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])
 
    data_set_init = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    data_set_init.reduce_to_active_class([41, 39])
    data_set_test.reduce_to_active_class([41, 39])
    
    keep_indices = [*range(args.subset)]
    data_set_init.reduce_to_active(keep_indices)

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
    resultstarget = {}
    correct = {}
    difficult_idx = {}

    for seed in tqdm(range(-1, N_SUBSET_ORDERING)):
        if seed != -1:
            seed = data_set.active_indices[seed]
        print("--------------------------", flush=True)
        params["training"]["num_epochs"] = params["training"]["num_epochs_init"]
        resultskl1_diag[seed] = {}
        resultskl2_diag[seed] = {}
        resultstarget[seed] = {}
        correct[seed] = {}

        params["model"]["seed"] = 43
        data_set = deepcopy(data_set_init)
        if seed != -1:
            data_set.remove_index_from_data(seed)
        train_loader_0 = torch.utils.data.DataLoader(
                data_set,
                batch_size=len(data_set), # params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
        criterion, (model_all, train_accuracy) = train_with_params(
            params,
            train_loader_0,
            test_loader
        )
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        N_REMOVE = len(data_set)
        for rem_idx in tqdm(range(N_REMOVE)):
            start_time = time.time()
            data_set_rm = deepcopy(data_set)
            data_set_rm.remove_curr_index_from_data(rem_idx)
            
            train_loader_rm = torch.utils.data.DataLoader(
                data_set_rm,
                batch_size=len(data_set_rm),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            _, (model_rem, _) = train_with_params(
                params,
                train_loader_rm,
                test_loader,
                rem = True
            )

            true_rm_idx = data_set.active_indices[rem_idx]
            
            
            if "diag" in representation:
                kl1_2, kl2_2, mean_diff_s, mean_diff_m, kl1_elem, kl2_elem = computeKL(DEVICE, backend_class, "diag", model_rem, model_all, train_loader_0, train_loader_0, params["training"]["l2_regularizer"], subset_of_weights=params["kllayer"]) 
                print(kl1_2)
                resultskl1_diag[seed][true_rm_idx] = kl1_2
                resultskl2_diag[seed][true_rm_idx] = kl2_2

            resultstarget[seed][true_rm_idx] = data_set.labels[rem_idx]

            correct_list = []
            for _, (data, target, idx, _) in enumerate(train_loader_0):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model_rem(data)
                _, predicted = torch.max(output.data, 1)
                correct_list.append((predicted == target).cpu().numpy())
            correct[seed][true_rm_idx] = correct_list

    if args.name != None:
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_infl/results_" + str(args.name)
    else:
        name_str = f"{params['model']['dataset_name']}_{params['model']['model']}_{N_SUBSET_ORDERING}_{args.epochs}_{args.subset}_{train_accuracy:.4f}_{params['training']['learning_rate']}{args.name_ext}"
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_infl/results_" + name_str

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    results_all = {
        "kl1_diag" : resultskl1_diag,
        "kl2_diag" : resultskl2_diag,
        "labels": resultstarget,
        "correct_data": correct,
        "difficult_idx": difficult_idx
    }
    with open(params["Paths"]["final_save_path"] + "/results_all.pkl", 'wb') as file:
        pickle.dump(results_all, file)
    print(f'Saving at {params["Paths"]["final_save_path"] + "/results_all.pkl"}')



