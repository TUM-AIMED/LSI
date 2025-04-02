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
            correct_pred = predicted==target
            correct += (predicted == target).sum().item()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            accuracy = correct/total
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # print(f"training accuracy {accuracy:.4f}, loss {loss:.4f}, epoch took {time.time() - start_time:.4f}")
            return np.mean(np.array(loss_list)), accuracy, correct_pred
        
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

    data = None
    target = None
    idx = None   
    if params["Full_Batch"]:
        data, target, idx, _ = next(iter(train_loader_0))
        data, target = data.to(DEVICE), target.to(DEVICE)

    correct_pred_list = []
    for epoch in range(params["training"]["num_epochs"]):
        # print(epoch)
        model.train()
        train_loss, train_accuracy, correct_pred = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0,
               data=data,
               target=target,
               idx=idx)
        if not rem:
            correct_pred_list.append(correct_pred)
        model.eval()
        if epoch % params["testing"]["test_every"] == 0:
            val_loss, val_accuracy = normal_val_step(params,
                model, 
                optimizer, 
                DEVICE, 
                criterion, 
                test_loader)
        if scheduler != None:
            scheduler.step()
    print(f"Epoch {epoch} with training_loss {train_loss} and val_loss {val_loss}")
    print(f"train accuracy {train_accuracy:.4f}")
    print(f"val accuracy {val_accuracy:.4f}")

    
    return model, train_accuracy, correct_pred_list


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
    print(f"Length = {N}")

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg4" or params["model"]["model"] == "logreg3":
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
    scheduler =    torch.optim.lr_scheduler.MultiStepLR(optimizer, [250, 350, 450, 550], gamma=0.8)



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
    parser.add_argument("--n_rem", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--repr", type=str, nargs='*', default=["diag"], help="Value for lerining_rate (optional)")
    parser.add_argument("--lap_type", type=str, default="asdlgnn", help="Value for lerining_rate (optional)")
    parser.add_argument("--freeze", type=bool, default=False, help="Value for lerining_rate (optional)")
    parser.add_argument("--epochs", type=int, default=8, help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="logreg4", help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--kllayer", type=str, default="all", help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=str, default="", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=1e-1, help="Value for lerining_rate (optional)")


    args = parser.parse_args()

    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
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
    params["testing"]["test_every"] = params["training"]["num_epochs_init"] -1
    params["Paths"] = {}
    params["Full_Batch"] = True

    params["freeze"] = args.freeze
    params["kllayer"] = args.kllayer

    
    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])
 
    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    class_list = [0, 1, 2]
    data_set.reduce_to_active_class(class_list)
    data_set_test.reduce_to_active_class(class_list)
    
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    random.seed(0)
    n = int(0.1 * N_REMOVE)
    random_labels_idx = []
    if args.corrupt == "noisy":
        random_labels_idx = random.sample([*range(N_REMOVE)], n)  # [0 + 5 * i for i in range(n) if i < args.subset/5] 
        data_set.apply_label_noise(random_labels_idx)
    elif args.corrupt == "artefact":
        random_labels_idx = random.sample([*range(N_REMOVE)], n)
        data_set.apply_image_mark(random_labels_idx)
    elif args.corrupt == "gnoisy":
        random_labels_idx = data_set.apply_group_label_noise()

    if N_REMOVE == 0:
        N_REMOVE = len(data_set.labels)

    # data_set_test._set_classes([0, 1])
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=len(data_set_test),
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
    kl1_elem_dict = {}
    kl2_elem_dict = {}
    correct = {}
    correct_pred_wo_rm = {}
    for seed in tqdm(range(N_SEEDS)):
        print("--------------------------", flush=True)
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
        kl1_elem_dict[seed] = {}
        kl2_elem_dict[seed] = {}
        correct[seed] = {}

        params["model"]["seed"] = seed

        train_loader_0 = torch.utils.data.DataLoader(
                data_set,
                batch_size=len(data_set), # params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
        criterion, (model_all, train_accuracy, correct_pred) = train_with_params(
            params,
            train_loader_0,
            test_loader
        )
        correct_pred_wo_rm[seed] = correct_pred
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        for rem_idx in range(N_REMOVE):
            start_time = time.time()
            print("--------------------------")
            print(f"Compute Remove Model {rem_idx}", flush=True)
            print("--------------------------")
            data_set_rm = deepcopy(data_set)
            data_set_rm.remove_curr_index_from_data(rem_idx)
            
            train_loader_rm = torch.utils.data.DataLoader(
                data_set_rm,
                batch_size=len(data_set_rm),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            print(f"removing datapoint took {time.time() - start_time}", flush=True)
            start_time = time.time()
            _, (model_rem, _, _) = train_with_params(
                params,
                train_loader_rm,
                test_loader,
                rem = True
            )
            print(f"training took {time.time() - start_time}", flush=True)

            true_rm_idx = data_set.active_indices[rem_idx]
            true_random_labels_idx = data_set.active_indices[random_labels_idx]

            
            start_time = time.time()
            
            # Replace second train_loader_0 with train_loader_rm to get matchingkl results
            if "kron" in representation:
                print("Compute Kron", flush=True)
                kl1_1, kl2_1, mean_diff_s, mean_diff_m, kl1_elem, kl2_elem = computeKL(DEVICE, backend_class, "kron", model_rem, model_all, train_loader_0, train_loader_0, params["training"]["l2_regularizer"], subset_of_weights=params["kllayer"]) 
                resultskl1_kron[seed][true_rm_idx] = kl1_1
                resultskl2_kron[seed][true_rm_idx] = kl2_1
            if "diag" in representation:
                print("Compute Diag", flush=True)
                kl1_2, kl2_2, mean_diff_s, mean_diff_m, kl1_elem, kl2_elem = computeKL(DEVICE, backend_class, "diag", model_rem, model_all, train_loader_0, train_loader_0, params["training"]["l2_regularizer"], subset_of_weights=params["kllayer"]) 
                resultskl1_diag[seed][true_rm_idx] = kl1_2
                resultskl2_diag[seed][true_rm_idx] = kl2_2
                print(f"kl divergence of {kl1_2}")
            if "full" in representation:
                print("Compute Full", flush=True)
                kl1_3, kl2_3, mean_diff_s, mean_diff_m, kl1_elem, kl2_elem = computeKL(DEVICE, backend_class, "full", model_rem, model_all, train_loader_0, train_loader_0, params["training"]["l2_regularizer"], subset_of_weights=params["kllayer"]) 
                resultskl1_full[seed][true_rm_idx] = kl1_3
                resultskl2_full[seed][true_rm_idx] = kl2_3
            print(f"kl computation took {time.time() - start_time}", flush=True)            

            mean_diff_sum[seed][true_rm_idx] = mean_diff_s
            mean_diff_mean[seed][true_rm_idx] = mean_diff_m
            kl1_elem_dict[seed][true_rm_idx] = kl1_elem
            kl2_elem_dict[seed][true_rm_idx] = kl2_elem
            resultstarget[seed][true_rm_idx] = data_set.labels[rem_idx]

            correct_list = []
            for _, (data, target, idx, _) in enumerate(train_loader_0):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model_rem(data)
                _, predicted = torch.max(output.data, 1)
                correct_list.append((predicted == target).cpu().numpy())
            correct[seed][true_rm_idx] = correct_list

    if args.name != None:
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script_final/results_" + str(args.name)
    else:
        name_str = f"{params['model']['dataset_name']}_{params['model']['model']}_{N_SEEDS}_{N_REMOVE}_{args.corrupt}_{args.epochs}_{args.subset}_{train_accuracy:.4f}_{params['training']['learning_rate']}{args.name_ext}"
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script_final/results_" + name_str

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    results_all = {
        "kl1_diag" : resultskl1_diag,
        "kl2_diag" : resultskl2_diag,
        "kl1_kron" : resultskl1_kron,
        "kl2_kron" : resultskl2_kron,
        "kl1_full" : resultskl1_full,
        "kl2_full" : resultskl2_full,
        "kl1_elem" : kl1_elem_dict,
        "kl2_elem" : kl2_elem_dict,
        "labels": resultstarget,
        "mean_diff_sum": mean_diff_sum,
        "mean_diff_mean": mean_diff_mean,
        "random_labels_idx": true_random_labels_idx,
        "correct_data": correct,
        "correct_pred": correct_pred_wo_rm
    }
    with open(params["Paths"]["final_save_path"] + "/results_all.pkl", 'wb') as file:
        pickle.dump(results_all, file)
    print(f'Saving at {params["Paths"]["final_save_path"] + "/results_all.pkl"}')

