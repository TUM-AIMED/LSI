import os
import pickle
import torch
from copy import deepcopy
from tqdm import tqdm
from LSI.experiments.utils_kl import _computeKL, _computeKL_from_full, _computeKL_from_kron
from LSI.models.models import get_model
from Datasets.dataset_helper import get_dataset
from utils import set_seed, set_deterministic
from LSI.experiments.utils_laplace import get_mean_and_prec
from utils_parser import get_parser
from utils_sam import SAM
from Datasets.dataset_wrappers import BatchDatasetWrapper
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_private(args, data_set, model, optimizer):
    train_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
    privacy_engine = PrivacyEngine()
    model = ModuleValidator.fix(model)
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    if args.make_private:
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = get_parser()
    parser.add_argument('--active_classes', type=int, nargs='+', default=[], help='List of active classes to reduce the dataset to')
    args = parser.parse_args()

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_full_vs_diag_vs_kfac"
    file_name = args.name or f"kl_jax_torch_{args.epochs}_remove_{args.n_rem}_dataset_{args.dataset}_model_{args.model}_laptype{args.laptype}_subset_{args.subset}_corrupt_{args.corrupt}_corrupt_data_{args.corrupt_data}_{args.corrupt_data_label}_{args.name_ext}"

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

   
    n_classes = len(torch.unique(data_set.labels))
    data_set = BatchDatasetWrapper(data_set, args.num_batches)

    # X_test, y_test = data_set_test.data.to(DEVICE), data_set_test.labels.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    model_class = get_model(args.model)
    kl, idx, pred, square_diff, losses_all = [], [], [], [], []

    for seed in range(args.n_seeds):
        set_seed(seed + 5)
        set_deterministic()
        model = model_class(n_classes, in_features=768 if args.dataset == "Imdbcompressed" else None).to(DEVICE)
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

        model, optimizer = make_private(args, data_set, model, optimizer)
        losses, models = training_loop(args, model, data_set, criterion, optimizer)
        losses_all.append(losses)

        means1, precs1 = [], []
        for model in range(models):
            mean1, prec1 = get_mean_and_prec(data_set, model, mode=args.laptype)
            means1.append(mean1)    
            precs1.append(prec1)    

        kl_seed, idx_seed = [], []
        for i in tqdm(range(args.range1, args.range2)):
            data_set_rem = deepcopy(data_set)
            if i in corrupted_idx:
                print("Corrupted index")
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

            model, optimizer = make_private(args, data_set_rem, model, optimizer)
            losses, models = training_loop(args, model, data_set_rem, criterion, optimizer)
            losses_all.append(losses)

            means2, precs2 = [], []
            for model in range(models):
                mean2, prec2 = get_mean_and_prec(data_set_rem, model, mode=args.laptype)
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

    result = {"idx": idx, "kl": kl, "pred": pred, "square_diff": square_diff, "corrupted_idx": corrupted_idx}
    os.makedirs(path_name, exist_ok=True)
    file_path = os.path.join(path_name, file_name + ".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    print(f'Saving at {file_path}')
