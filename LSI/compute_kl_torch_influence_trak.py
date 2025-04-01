import numpy as np
import torch
from torch.utils.data import TensorDataset
from Datasets.dataset_helper import get_dataset
import os
import pickle
import argparse
import gc
from models.final_models_to_test import get_model
import random 
from pydvl.influence.torch import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from trak import TRAKer

gc.collect()
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

# Set deterministic and disable benchmarking
def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=3, help="Value for lerining_rate (optional)")
    parser.add_argument("--epochs", type=int, default=10, help="Value for lerining_rate (optional)") # 150 for Resnet, 700 for CNN, 600 for MLP
    parser.add_argument("--lr", type=float, default=0.004, help="Value for lerining_rate (optional)") # General 0.004 0.01 for Resnet, 0.005 for CNN and MLP
    parser.add_argument("--mom", type=float, default=0.9, help="Value for lerining_rate (optional)") # 0.9

    parser.add_argument("--dataset", type=str, default="Imagenette", help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="ResNet18", help="Value for lerining_rate (optional)")
 

    args = parser.parse_args()

    mom = args.mom
    wd = 5e-4 # 0.01

    path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_Influence"
    file_name = f"{args.dataset}_{args.model}_trak"

    data_set_class, data_path = get_dataset(args.dataset)

    if args.dataset == "cifar10":
        data_set = data_set_class(data_path, train=True, ret4=False)
        data_set_test = data_set_class(data_path, train=False, ret4=False) 
    else:
        data_set = data_set_class(data_path, train=True)
        data_set_test = data_set_class(data_path, train=False) 


    train_data = DataLoader(data_set, batch_size=256)
    train_data2 = DataLoader(data_set, batch_size=1)
    test_data = DataLoader(data_set_test, batch_size=256)
    X_train = []
    y_train = []
    X_test = []  
    y_test = []  
    for data, target in tqdm(train_data):
        X_train.append(data)
        y_train.append(target)
    for data, target in tqdm(test_data):
        X_test.append(data)
        y_test.append(target)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    n_classes = len(torch.unique(y_test))
    model_class = get_model(args.model)
    train_data2 = DataLoader(TensorDataset(X_train.cpu(), y_train.cpu()), batch_size=256)
    test_data = DataLoader(TensorDataset(X_test.cpu(), y_test.cpu()), batch_size=256)

    influences_all = []
    N_SEEDS = args.n_seeds
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Initialize variables
influences_all = []

checkpoints = []
for seed in tqdm(range(N_SEEDS)):
    set_seed(seed + 5)
    set_deterministic()
    model = model_class(n_classes)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=wd, momentum=mom, nesterov=True)
    
    losses = []
    for i in tqdm(range(args.epochs)):
        for data, target in train_data:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            pred = torch.argmax(output, dim=1)
            accuracy = torch.sum(pred == target) / len(target)
            print(accuracy)
            loss = criterion(output, target)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    # Save the trained model checkpoint after training
    checkpoints.append(model.state_dict())

# Initialize TRAKer for data valuation
loader_train = train_data2  # Replace with your train dataloader

traker = TRAKer(model=model,
                task='image_classification',
                train_set_size=len(loader_train.dataset),
                proj_dim=4096)  # You can adjust the projection dimension as needed

# Load the trained checkpoint for TRAKer
for model_id, ckpt in enumerate(tqdm(checkpoints)):
    traker.load_checkpoint(ckpt, model_id=model_id)

    # Featurize the training data (train set) for TRAKer
    for batch in tqdm(loader_train):
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

# Finalize the features after processing all training data
traker.finalize_features()

# Now, we will score the test data for TRAK scores
loader_test = loader_train  # Replace with your test dataloader
for model_id, ckpt in enumerate(tqdm(checkpoints)):
    traker.start_scoring_checkpoint(exp_name='quickstart',
                                    checkpoint=ckpt,
                                    model_id=model_id,
                                    num_targets=len(loader_test.dataset))
    for batch in loader_test:
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

scores = traker.finalize_scores(exp_name='quickstart')



# Save the TRAK scores
result = {"trak_scores": scores}
if not os.path.exists(path_name):
    os.makedirs(path_name)
file_path = os.path.join(path_name, file_name + ".pkl")
with open(file_path, 'wb') as file:
    pickle.dump(result, file)
print(f'Saving at {file_path}')