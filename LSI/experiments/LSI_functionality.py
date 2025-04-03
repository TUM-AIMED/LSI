import os
import pickle
import torch
from copy import deepcopy
from tqdm import tqdm
from LSI.utils.utils_kl import _computeKL, _computeKL_from_full, _computeKL_from_kron
from LSI.models.models import get_model
from LSI.Datasets.dataset_helper import get_dataset
from LSI.utils.utils import set_seed, set_deterministic
from LSI.utils.utils_laplace import get_mean_and_prec
from LSI.utils.utils_parser import get_parser, generate_experiment_description
from LSI.utils.utils_sam import SAM
from LSI.Datasets.dataset_wrappers import BatchDatasetWrapper
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def make_private(args, data_set, model, optimizer):
    
    train_loader = DataLoader(data_set, batch_size=len(data_set.data[0]), shuffle=False)
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    privacy_engine = PrivacyEngine()
    model = ModuleValidator.fix(model)
    model, optimizer, _ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise,
                max_grad_norm=args.clip,
                noise_generator = generator,
                loss_reduction = "mean"
            )
    return model, optimizer

def training_loop(args, model, data_set, criterion, optimizer):
    model = model.to(DEVICE)
    model.train()
    # Stack X_train_batches and y_train_batches into tensors
    max_len = max(len(batch) for batch in data_set.data)
    X_train_batches = [torch.cat([batch, torch.zeros(max_len - len(batch), *batch.shape[1:])], dim=0) if len(batch) < max_len else batch for batch in data_set.data]
    y_train_batches = [torch.cat([batch, torch.full((max_len - len(batch),), -1)]) if len(batch) < max_len else batch for batch in data_set.labels]

    X_train_batches = torch.stack(X_train_batches).to(DEVICE)
    y_train_batches = torch.stack(y_train_batches).to(DEVICE)

    losses = []
    models = []
    for epoch in range(args.epochs):
        for X_train, y_train in zip(X_train_batches, y_train_batches):
            # Remove dummy values
            valid_indices = y_train != -1
            X_train, y_train = X_train[valid_indices], y_train[valid_indices]
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
        if epoch == args.epochs - 1:
            models.append(deepcopy(model.cpu()))
    return losses, models


def run_experiment(args, data_set):
    # data_set_test = data_set_class(data_path, train=False)
    corrupted_idx = []
    if args.human_label_noise:
        print("Corrupting labels with human-annotated labels")
        corrupted_idx = data_set.apply_human_label_noise(noise_file_path=args.noisy_label_location, dname=args.dataset)
    if args.corrupt_label > 0:
        corrupted_idx = data_set.corrupt_label(args.corrupt_label)
    if args.corrupt_data > 0:
        corrupted_idx = data_set.corrupt_data(args, args.corrupt_data, args.corrupt_data_label)
    if len(args.active_classes) > 0: 
        data_set.filter_by_classes(args.active_classes)
    if args.subset > 0:
        data_set.filter_by_indices(list(range(args.subset)))
    corrupted_idx = [idx for idx in corrupted_idx if idx in data_set.active_indices]

    generate_experiment_description(args)

    args.range1 = args.range1 if args.range1 > 0 else 0
    args.range2 = args.range2 if args.range2 > 0 else len(data_set)
   
    n_classes = len(torch.unique(data_set.labels))
    data_set = BatchDatasetWrapper(data_set, args.num_batches)

    # X_test, y_test = data_set_test.data.to(DEVICE), data_set_test.labels.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    model_class = get_model(args.model)
    kl, idx, square_diff, losses_all = [], [], [], []

    for seed in range(args.n_seeds):
        set_seed(seed + 5)
        set_deterministic()
        model = model_class(n_classes, in_features=768 if args.dataset == "Imdbcompressed" else 512).to(DEVICE)
        backup_model = deepcopy(model)
        if args.use_sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(
                model.parameters(),
                base_optimizer=base_optimizer,
                lr=args.lr,
                momentum=args.mom
            )
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom, nesterov=True)
        if args.make_private:
            model, optimizer = make_private(args, data_set, model, optimizer)
        losses, models = training_loop(args, model, data_set, criterion, optimizer)
        losses_all.append(losses)

        means1, precs1 = [], []
        for model in models:
            mean1, prec1 = get_mean_and_prec(data_set, model, mode=args.laptype, DEVICE=DEVICE)
            means1.append(mean1)    
            precs1.append(prec1)    

        kl_seed, idx_seed = [], []
        for i in tqdm(range(args.range1, args.range2)):
            data_set_rem = deepcopy(data_set)
            data_set_rem.remove_index(i)
            model = deepcopy(backup_model).to(DEVICE)
            if args.use_sam:
                optimizer = SAM(
                    model.parameters(),
                    base_optimizer=base_optimizer,
                    lr=args.lr,
                    momentum=args.mom
                )
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom, nesterov=True)
            if args.make_private:
                model, optimizer = make_private(args, data_set_rem, model, optimizer)
            losses, models = training_loop(args, model, data_set_rem, criterion, optimizer)
            losses_all.append(losses)

            means2, precs2 = [], []
            for model in models:
                mean2, prec2 = get_mean_and_prec(data_set_rem, model, mode=args.laptype, DEVICE=DEVICE)
                means2.append(mean2)    
                precs2.append(prec2)

            kl1s, square_diff1s = [], []
            for mean1, mean2, prec1, prec2 in zip(means1, means2, precs1, precs2):
                if args.laptype == "full":
                    kl1, square_diff1 = _computeKL_from_full(mean1, mean2, prec1, prec2)
                elif args.laptype == "diag":
                    kl1, square_diff1 = _computeKL(mean1, mean2, prec1, prec2)
                elif args.laptype == "kfac":
                    kl1, square_diff1 = _computeKL_from_kron(mean1, mean2, prec1, prec2)
                else:
                    raise Exception("Not implemented")
                kl1s.append(kl1)
                square_diff1s.append(square_diff1)

            kl_seed.append(kl1s)
            square_diff.append(square_diff1s)
            idx_seed.append(i)

        kl.append(kl_seed)
        idx.append(idx_seed)

    result = {"idx": idx, "kl": kl, "square_diff": square_diff, "corrupted_idx": corrupted_idx}
    file_name = f"dataset_{args.dataset}_subset_{args.subset}_model_{args.model}_epochs_{args.epochs}_remove_{args.range1}_to_{args.range2}_laptype{args.laptype}_corrupt_label_{str(args.corrupt_label).replace('.', '')}_corrupt_data_{str(args.corrupt_data).replace('.', '')}_{args.corrupt_data_label}"
    os.makedirs(os.path.join(args.save_dir, args.sub_dir), exist_ok=True)
    file_path = os.path.join(args.save_dir, args.sub_dir, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')