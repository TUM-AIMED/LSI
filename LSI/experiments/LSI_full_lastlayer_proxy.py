from Datasets.dataset_helper import get_dataset
from LSI.experiments.utils_kl import _computeKL, _computeKL_from_full, _computeKL_from_kron
import torch
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
from LSI.models.models import get_model
from utils import set_seed, set_deterministic
from LSI.experiments.utils_laplace import get_mean_and_prec
from utils_parser import get_parser
from Datasets.dataset_compressed import CompressedDataset
from Datasets.dataset_wrappers import BatchDatasetWrapper
from ..models.model_wrapper import CompressionWrapper, ProxyWrapper

def training_loop(args, model, data_set, criterion, optimizer):
    model=model.to(DEVICE)
    X_train_batches, y_train_batches = data_set.data.to(DEVICE), data_set.labels.to(DEVICE)
    losses = []
    models = []
    for epoch in range(args.epochs):
        for X_train, y_train in zip(X_train_batches, y_train_batches):
            if args.use_sam:
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        if args.kl_every_n_epochs > 0 and epoch % args.kl_every_n_epochs == 0:
            models.append(deepcopy(model.cpu()))
    return losses, models


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = get_parser()
    args = parser.parse_args()

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_all_layers_vs_last_layer"
    file_name = "full_vs_lastlayer_SmallCNN_all_three_500_convex"

    data_set_class, data_path = get_dataset(args.dataset)

    data_set_class, data_path = get_dataset(args.dataset)
    data_set = data_set_class(data_path, train=True)
    # data_set_test = data_set_class(data_path, train=False)
    corrupted_idx = None
    if len(args.active_classes) > 0: 
        data_set.reduce_to_active_class(args.active_classes)
    if args.subset > 0:
        data_set.reduce_to_active(range(args.subset))
    if args.human_label_noise:
        print("Corrupting labels with human-annotated labels")
        corrupted_idx = data_set.apply_human_label_noise()
    if args.corrupt > 0:
        corrupted_idx, y_train = data_set.corrupt_label(args.corrupt)
    if args.corrupt_data > 0:
        if args.dataset == "Imdbcompressed":
            corrupted_idx = data_set.corrupt_data(args, args.corrupt_data, args.corrupt_data_label)
        else:
            corrupted_idx = data_set.corrupt_data(args, args.corrupt_data, args.corrupt_data_label)

    args.range1 = args.range1 if args.range1 > 0 else 0
    args.range2 = args.range2 if args.range2 > 0 else len(data_set)
    args.n_rem = args.n_rem if args.n_rem > 0 else len(data_set)

    # X_test, y_test = data_set_test.data.to(DEVICE), data_set_test.labels.to(DEVICE)
    n_classes = len(torch.unique(data_set.labels))
    data_set = BatchDatasetWrapper(data_set, args.num_batches)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    i = 0
    kl = []
    kl_ll = []
    idx = []
    pred = []
    square_diff = []
    kl_proxy = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    set_seed(5)
    set_deterministic()

    model_class = get_model(args.model)
    model = model_class(n_classes)
    backup_model = deepcopy(model)
    model = deepcopy(backup_model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd, momentum=args.mom, nesterov=True)
    
    losses, models = training_loop(args, model, data_set, criterion, optimizer)

    means1, precs1 = [], []
    means1_ll, precs1_ll = [], []
    for model in models:
        if args.laptype == "full":
            mean1, prec1 = get_mean_and_prec(data_set, model, mode="full")
            mean1_ll, prec1_ll = get_mean_and_prec(data_set, model, mode="full", subset="last_layer")
        elif args.laptype == "diag":
            mean1, prec1 = get_mean_and_prec(data_set, model, mode="diag")
            mean1_ll, prec1_ll = get_mean_and_prec(data_set, model, mode="diag", subset="last_layer")
        elif args.laptype == "kfac":
            mean1, prec1 = get_mean_and_prec(data_set, model, mode="kfac")
            mean1_ll, prec1_ll = get_mean_and_prec(data_set, model, mode="kfac", subset="last_layer")
        else:
            raise Exception("Not implemented")
        
        means1.append(mean1)
        precs1.append(prec1)
        means1_ll.append(mean1_ll)
        precs1_ll.append(prec1_ll)

    #####################################
    #####################################
    #####################################
    compression_model = CompressionWrapper(deepcopy(model))
    data_compressed, labels = compression_model.compress(data_set).detach()
    dataset_compressed = CompressedDataset(data_compressed, labels)

    model_proxy = ProxyWrapper(deepcopy(backup_model))

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    torch.manual_seed(5)

    
    backup_model_proxy = deepcopy(model_proxy)
    optimizer_proxy = torch.optim.SGD(
        model_proxy.parameters(),
        lr=args.lr2,
        weight_decay=args.wd, momentum=args.mom, nesterov=True)
    losses, models = training_loop(args, model_proxy, dataset_compressed, criterion, optimizer_proxy)

    means1_proxy, precs1_proxy = [], []
    for model_proxy in models:
        if args.laptype == "full":
            mean1_proxy, prec1_proxy = get_mean_and_prec(dataset_compressed, model_proxy, mode="full")
        elif args.laptype == "diag":
            mean1_proxy, prec1_proxy = get_mean_and_prec(dataset_compressed, model_proxy, mode="diag")
        elif args.laptype == "kfac":
            mean1_proxy, prec1_proxy = get_mean_and_prec(dataset_compressed, model_proxy, mode="kfac")
        else:
            raise Exception("Not implemented")
        
        means1_proxy.append(mean1_proxy)
        precs1_proxy.append(prec1_proxy)




    kl_seed = []
    kl_seed_ll = []
    idx_seed = []
    pred_seed = []
    kl_seed_proxy = []

    for i in tqdm(range(args.range1, args.range2)):
        data_set_rem = deepcopy(data_set)
        data_set_rem.remove_index(i)
        dataset_compressed_rem = deepcopy(dataset_compressed)
        dataset_compressed_rem.remove_index(i)

        model = deepcopy(backup_model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd, momentum=args.mom, nesterov=True)

        model_proxy = deepcopy(backup_model_proxy)
        optimizer_proxy = torch.optim.SGD(
            model_proxy.parameters(),
            lr=args.lr2,
            weight_decay=args.wd, momentum=args.mom, nesterov=True)


        losses, models = training_loop(args, model, data_set_rem, criterion, optimizer)
        losses, models_proxy = training_loop(args, model_proxy, dataset_compressed_rem, criterion, optimizer_proxy)

        means2, precs2 = [], []
        means2_ll, precs2_ll = [], []
        means2_proxy, precs2_proxy = [], []

        for model, model_proxy in zip(models, models_proxy):
            if args.laptype == "full":
                mean2, prec2 = get_mean_and_prec(data_set_rem, model, mode="full")
                mean2_ll, prec2_ll = get_mean_and_prec(data_set_rem, model, mode="full", subset="last_layer")
                mean2_proxy, prec2_proxy = get_mean_and_prec(dataset_compressed_rem, model_proxy, mode="full")
            elif args.laptype == "diag":
                mean2, prec2 = get_mean_and_prec(data_set_rem, model, mode="diag")
                mean2_ll, prec2_ll = get_mean_and_prec(data_set_rem, model, mode="diag", subset="last_layer")
                mean2_proxy, prec2_proxy = get_mean_and_prec(dataset_compressed_rem, model_proxy, mode="diag")
            elif args.laptype == "kfac":
                mean2, prec2 = get_mean_and_prec(data_set_rem, model, mode="kfac")
                mean2_ll, prec2_ll = get_mean_and_prec(data_set_rem, model, mode="kfac", subset="last_layer")
                mean2_proxy, prec2_proxy = get_mean_and_prec(dataset_compressed_rem, model_proxy, mode="kfac")
            else:
                raise Exception("Not implemented")

            means2.append(mean2)
            precs2.append(prec2)
            means2_ll.append(mean2_ll)
            precs2_ll.append(prec2_ll)
            means2_proxy.append(mean2_proxy)
            precs2_proxy.append(prec2_proxy)

            kl1_list = []
            kl1_ll_list = []
            kl1_proxy_list = []
            square_diff1_list = []

            for (mean1, mean2, prec1, prec2, 
                 mean1_ll, mean2_ll, prec1_ll, prec2_ll, 
                 mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy) in zip(
                 means1, means2, precs1, precs2, 
                 means1_ll, means2_ll, precs1_ll, precs2_ll, 
                 means1_proxy, means2_proxy, precs1_proxy, precs2_proxy):
                if args.laptype == "diag":
                    kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
                    kl1_ll, _ = _computeKL(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
                    kl1_proxy, _ = _computeKL(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
                elif args.laptype == "full":
                    kl1, square_diff1 = _computeKL_from_full(mean1, mean2, prec1, prec2)
                    kl1_ll, _ = _computeKL_from_full(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
                    kl1_proxy, _ = _computeKL_from_full(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
                elif args.laptype == "kfac":
                    kl1, square_diff1 = _computeKL_from_kron(mean1, mean2, prec1, prec2)
                    kl1_ll, _ = _computeKL_from_kron(mean1_ll, mean2_ll, prec1_ll, prec2_ll)
                    kl1_proxy, _ = _computeKL_from_kron(mean1_proxy, mean2_proxy, prec1_proxy, prec2_proxy)
                else:
                    raise Exception("Not implemented")

                kl1_list.append(kl1)
                kl1_ll_list.append(kl1_ll)
                kl1_proxy_list.append(kl1_proxy)
                square_diff1_list.append(square_diff1)


        print(f"KL divergence {kl1_list}")
        print(f"KL divergence_ll {kl1_ll_list}")
        print(f"KL divergence_proxy {kl1_proxy_list}")
        kl_seed.append(kl1_list)
        kl_seed_ll.append(kl1_ll_list)
        kl_seed_proxy.append(kl1_proxy_list)
        square_diff.append(square_diff1_list)
        idx_seed.append(i)
    kl_proxy.append(kl_seed_proxy)
    kl.append(kl_seed)
    kl_ll.append(kl_seed_ll)
    idx.append(idx_seed)

    result = {"idx": idx,
              "kl_full": kl,
              "kl_ll": kl_ll,
              "kl_proxy": kl_proxy,
              "pred":pred, 
              "square_diff":square_diff}
    

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')