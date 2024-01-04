import sys
import os
import pickle

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from opacus.validators.module_validator import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from opacus.privacy_engine import PrivacyEngine
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import warnings
import argparse
import time
import random
from laplace.curvature import AsdlGGN, BackPackGGN, AsdlHessian, AsdlEF, BackPackEF
from utils.kl_div import computeKL
from collections import defaultdict



from utils.plot_utils import plot_and_save_histogram, plot_and_save_lineplot, plot_and_save_lineplot_with_running_sum


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def log_realized_gradients(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms):
    """
    train runs the epoch loop and calls train_step and test

    :param params: params dict
    :param model: pytorch model
    :param DEVICE: available device
    :param optimizer: optimizer fct (Adam, AdamW etc.)
    :param criterion: loss function
    :param train_loader_0: dataloader of the whole, unfiltered dataset
    :param grad_norms: tensor to save all the accumulated gradient norms - used for logging
    :param budget: privacy budget of data elements
    :param N: number of training samples

    """ 

    # Train loader returns also indices (vector idx)
    with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=500, optimizer=optimizer) as train_loader_0_new:
        for _, (data, target, idx, _) in enumerate(train_loader_0_new):
            # print(idx[0])
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward(retain_graph=True)

            batch_grad_norms = torch.zeros(len(target)).to(DEVICE)
            # Clip each parameter's per-sample gradient
            for (ii, p) in enumerate(model.parameters()):
                if not p.requires_grad:
                    continue
                per_sample_grad = p.grad_sample

                # dimension across which we compute the norms for this gradient part
                # (here is difference e.g. between biases and weight matrices)
                per_sample_grad = torch.reshape(per_sample_grad, (per_sample_grad.shape[0], per_sample_grad.shape[1], -1))
                dims = list(range(1, len(per_sample_grad.shape)))

                # compute the clipped norms. Gradients will be clipped in .backward()
                per_sample_grad_norms = per_sample_grad.norm(dim=dims)
                batch_grad_norms += per_sample_grad_norms ** 2


            # compute the clipped norms. Gradients will be then clipped in .backward()
            # clipped, per sample gradient norms, to track privacy
            for index, batch_grad_norm in zip(idx, batch_grad_norms):
                grad_norms[index.item()].append(((
                    torch.sqrt(batch_grad_norm).clamp(max=params["DP"]["max_per_sample_grad_norm"])
                ) ** 2).item())

    del batch_grad_norms
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    return 


def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_active):
   
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    with BatchMemoryManager(data_loader=train_loader_active, max_physical_batch_size=500, optimizer=optimizer) as train_loader_new:
        correct = 0
        total = 0
        loss_list = []
        for _, (data, target, idx, _) in enumerate(train_loader_new):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted.cpu() == target.cpu()).sum().item()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    print(f"accuracy was {accuracy}")
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
    rem = False,
    DP = False
):

    grad_norms = defaultdict(list)
    for epoch in range(params["training"]["num_epochs"]):
        # print(epoch)
        model.train()
        # if DP:
        #     log_realized_gradients(params,
        #                     model,
        #                     DEVICE,
        #                     optimizer,
        #                     criterion,
        #                     train_loader_0,
        #                     grad_norms)
        train_loss, train_accuracy = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0)
        model.eval()
        if epoch % params["testing"]["test_every"] == 0:
            val_loss, val_accuracy = normal_val_step(params,
                model, 
                optimizer, 
                DEVICE, 
                criterion, 
                test_loader)
    print(f"Epoch {epoch} with training_loss {train_loss} and val_loss {val_loss}")
    print(f"train accuracy {train_accuracy:.4f}")
    print(f"val accuracy {val_accuracy:.4f}")

    
    return model, train_accuracy, grad_norms


def train_with_params(
    params,
    train_loader_0,
    test_loader,
    rem = False,
    DP = False
):
    """
    train_with_params initializes the main training parts (model, criterion, optimizer and makes private)
    :param params: dict with all parameters
    """ 

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
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg3" or params["model"]["model"] == "logreg4" or params["model"]["model"] == "logreg5":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES))
        model = model.to(DEVICE)
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

    privacy_engine = PrivacyEngine(secure_mode=False)

    model, optimizer, train_loader_0 = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader_0,
        noise_multiplier=params["DP"]["sigma_tilde"],
        max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
        poisson_sampling=False,
    )


    return criterion, privacy_engine, train(
                                            params,
                                            model,
                                            DEVICE,
                                            train_loader_0,
                                            test_loader,
                                            optimizer,
                                            criterion,
                                            scheduler = scheduler,
                                            rem=rem,
                                            DP=DP
                                        )




if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_rem", type=int, default=3, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_seeds", type=int, default=3, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--repr", type=str, nargs='*', default=["diag"], help="Value for lerining_rate (optional)")
    parser.add_argument("--lap_type", type=str, default="asdlgnn", help="Value for lerining_rate (optional)")
    parser.add_argument("--freeze", type=bool, default=False, help="Value for lerining_rate (optional)")
    parser.add_argument("--epochs", type=int, default=200, help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="logreg4", help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar10compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--kllayer", type=str, default="all", help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=str, default="", help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=int, default=50000, help="Value for lerining_rate (optional)")
    parser.add_argument("--lr", type=float, default=5e-0, help="Value for lerining_rate (optional)")
    parser.add_argument("--clip", type=float, default=0.1, help="Value for lerining_rate (optional)")


    args = parser.parse_args()

    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    if args.epochs == 200:
        DP_RANGES = [1e-1, 4.4, 9.1, 13.5, 57]
    elif args.epochs == 100:
        DP_RANGES = [1e-1, 3.1, 6.5, 9.6, 41]
    else:
        raise Exception("no DP values for this number of epochs")

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
    params["model"]["seed"] = 42
    params["model"]["model"] = args.model
    params["model"]["dataset_name"] = args.dataset
    params["training"] = {}
    # params["training"]["batch_size"] = 50
    params["training"]["learning_rate"] = args.lr # 1e-2 #3e-1 for SGD on cifar100 # 6e-03 #cifar100# 2e-3# cifar10 logreg2/ cifar100 logreg3 1k/10k# 4e-02# 0.002 mlp #3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-8
    params["training"]["num_epochs_init"] = args.epochs
    params["testing"] = {}
    params["testing"]["test_every"] = params["training"]["num_epochs_init"] + 1
    params["Paths"] = {}
    params["Full_Batch"] = True
    params["DP"] = {}
    params["DP"]["sigma_tilde"] = 0
    params["DP"]["max_per_sample_grad_norm"] = args.clip
    params["freeze"] = args.freeze
    params["kllayer"] = args.kllayer

    
    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])
 
    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    # data_set.reduce_to_active_class([41, 39, 22, 20, 9])
    # data_set_test.reduce_to_active_class([41, 39, 22, 20, 9])
    
    keep_indices = [*range(args.subset)]
    data_set.reduce_to_active(keep_indices)
    random.seed(0)
    n = int(0.4 * N_REMOVE)
    random_labels_idx = []
    if args.corrupt == "noisy":
        random_labels_idx = [0 + 5 * i for i in range(n) if i < args.subset/5] # random.sample([*range(N_REMOVE)], n)
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
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    params_dict = {}
    resultskl1_diag = {}
    resultskl2_diag = {}
    resultstarget = {}
    mean_diff_sum = {}
    mean_diff_mean = {}
    kl1_elem_dict = {}
    kl2_elem_dict = {}
    correct = {}
    grad_norms_res = {}
    grad_norms_init = {}
    epsilon_init = {}
    epsilon_rem = {}
    for seed in tqdm(range(N_SEEDS)):
        params_dict[seed] = {}
        resultskl1_diag[seed] = {}
        resultskl2_diag[seed] = {}
        resultstarget[seed] = {}
        mean_diff_sum[seed] = {}
        mean_diff_mean[seed] = {}
        kl1_elem_dict[seed] = {}
        kl2_elem_dict[seed] = {}
        correct[seed] = {}
        grad_norms_res[seed] = {}
        grad_norms_init[seed] = {}
        epsilon_init[seed] = {}
        epsilon_rem[seed] = {}

        params["model"]["seed"] = seed
        for dp in tqdm(DP_RANGES):
            params["DP"]["sigma_tilde"] = dp
            print("--------------------------", flush=True)
            params["training"]["num_epochs"] = params["training"]["num_epochs_init"]
            resultskl1_diag[seed][dp] = {}
            resultskl2_diag[seed][dp] = {}
            resultstarget[seed][dp] = {}
            mean_diff_sum[seed][dp] = {}
            mean_diff_mean[seed][dp] = {}
            kl1_elem_dict[seed][dp] = {}
            kl2_elem_dict[seed][dp] = {}
            correct[seed][dp] = {}
            grad_norms_res[seed][dp] = {}
            grad_norms_init[seed][dp] = {}
            epsilon_rem[seed][dp] = {}

            train_loader_0 = torch.utils.data.DataLoader(
                    data_set,
                    batch_size=len(data_set), # params["training"]["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                
            criterion, privacy_engine, (model_all, train_accuracy, grad_norms_init_res) = train_with_params(
                params,
                train_loader_0,
                test_loader,
                DP = True
            )

            epsilon_init[seed][dp] = [privacy_engine.get_epsilon(delta) for delta in [1e-7, 1e-5, 1e-4, 1e-3]]
            train_X_example, _, _, _ = train_loader_0.dataset[0]
            N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))
            model_class = get_model(params["model"]["model"])
            if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
                model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
            else:
                model = model_class(len(train_X_example), N_CLASSES)
                model = model.to(DEVICE)
            model.load_state_dict(model_all._module.state_dict())
            model_all = model 
        
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
                _, privacy_engine, (model_rem, _, grad_norms) = train_with_params(
                    params,
                    train_loader_rm,
                    test_loader,
                    rem = True
                )
                print(f"training took {time.time() - start_time}", flush=True)
                train_X_example, _, _, _ = train_loader_0.dataset[0]
                N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))
                model_class = get_model(params["model"]["model"])
                if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
                    model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
                else:
                    model = model_class(len(train_X_example), N_CLASSES)
                    model = model.to(DEVICE)
                model.load_state_dict(model_rem._module.state_dict()) 
                model_rem = model
                true_rm_idx = data_set.active_indices[rem_idx]
                true_random_labels_idx = data_set.active_indices[random_labels_idx]

                
                start_time = time.time()
                

                if "diag" in representation:
                    print("Compute Diag", flush=True)
                    kl1_2, kl2_2, mean_diff_s, mean_diff_m, kl1_elem, kl2_elem = computeKL(DEVICE, backend_class, "diag", model_rem, model_all, train_loader_0, train_loader_0, params["training"]["l2_regularizer"], subset_of_weights=params["kllayer"]) 
                    resultskl1_diag[seed][dp][true_rm_idx] = kl1_2
                    resultskl2_diag[seed][dp][true_rm_idx] = kl2_2
                    print(kl1_2)

                mean_diff_sum[seed][dp][true_rm_idx] = mean_diff_s
                mean_diff_mean[seed][dp][true_rm_idx] = mean_diff_m
                kl1_elem_dict[seed][dp][true_rm_idx] = kl1_elem
                kl2_elem_dict[seed][dp][true_rm_idx] = kl2_elem
                resultstarget[seed][dp][true_rm_idx] = data_set.labels[rem_idx]
                grad_norms_res[seed][dp][true_rm_idx] = grad_norms

                epsilon_rem[seed][dp][true_rm_idx] = [privacy_engine.get_epsilon(delta) for delta in [1e-7, 1e-8, 1e-5, 1e-4, 1e-3]]

                correct_list = []
                for _, (data, target, idx, _) in enumerate(train_loader_0):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model_rem(data)
                    _, predicted = torch.max(output.data, 1)
                    correct_list.append((predicted == target).cpu().numpy())
                correct[seed][dp][true_rm_idx] = correct_list
            grad_norms_init[seed][dp] = grad_norms_init_res
            params_dict[seed][dp] = params


    if args.name != None:
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_w_DP/results_" + str(args.name)
    else:
        name_str = f"{params['model']['dataset_name']}_{params['model']['model']}_DP_Range_m09_1_{N_SEEDS}_{N_REMOVE}_{args.clip}_{args.epochs}_{args.subset}_{train_accuracy:.4f}_{params['training']['learning_rate']}{args.name_ext}"
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_w_DP/results_" + name_str

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    results_all = {
        "kl1_diag" : resultskl1_diag,
        "kl2_diag" : resultskl2_diag,
        "kl1_elem" : kl1_elem_dict,
        "kl2_elem" : kl2_elem_dict,
        "labels": resultstarget,
        "mean_diff_sum": mean_diff_sum,
        "mean_diff_mean": mean_diff_mean,
        "random_labels_idx": true_random_labels_idx,
        "correct_data": correct,
        "grad_norms": grad_norms_res,
        "grad_norms_init": grad_norms_init,
        "epsilon_init": epsilon_init,
        "epsilon_rem": epsilon_rem,
        "params": params_dict
    }
    with open(params["Paths"]["final_save_path"] + "/results_all.pkl", 'wb') as file:
        pickle.dump(results_all, file)
    print(f'Saving at {params["Paths"]["final_save_path"] + "/results_all.pkl"}')

