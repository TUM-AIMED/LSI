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


def training_loop(epochs, model, X_train, y_train, criterion, optimizer):
    model = model.to(DEVICE)
    for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
    return model

def extract_compressed_features_and_labels(dataloader, model):
    features = []
    labels = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # Pass through the model except the last layer
            all_layers = list(model.children())[:-1]  # Exclude the last layer
            feature_extractor = torch.nn.Sequential(*all_layers)
            feature = feature_extractor(X_batch)
            features.append(feature.cpu())
            labels.append(y_batch.cpu())
    return features, labels

def get_last_layer_model(model):
    """
    Returns a new model consisting only of the last layer of the given model.
    """
    last_layer = list(model.children())[-1]  # Extract the last layer
    last_layer_model = torch.nn.Sequential(last_layer)  # Wrap it in a Sequential container
    return last_layer_model
    
def compute_lsi_divergence(dataloader, model, criterion, laptype, epochs, lr, wd, mom, pretrain_settings, pretrain=True, seed=42):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    set_deterministic()

    # Load dataset and model
    criterion_pretrain = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer_pretrain = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom, nesterov=True)

    if pretrain:
        for epoch in pretrain_settings['epochs']:
            for X_train, y_train in dataloader:
                optimizer_pretrain.zero_grad()
                output = model(X_train)
                loss = criterion_pretrain(output, y_train)
                loss.backward()
                optimizer_pretrain.step()

    X_train, y_train = extract_compressed_features_and_labels(dataloader, model)
    proxy_model_init = get_last_layer_model(model)

    proxy_model = deepcopy(proxy_model_init)
    proxy_model = proxy_model.to(DEVICE)
    optimizer = torch.optim.SGD(proxy_model.parameters(), lr=lr, weight_decay=wd, momentum=mom, nesterov=True)
    model = training_loop(epochs, proxy_model, X_train, y_train, criterion, optimizer)
    data_set = CompressedDataset(X_train, y_train)
    mean1, prec1 = get_mean_and_prec(data_set, model, mode=laptype)

    kl_values = []

    for index in X_train:
        X_train_rem = torch.cat([X_train[0:index], X_train[index+1:]])
        y_train_rem = torch.cat([y_train[0:index], y_train[index+1:]])
        proxy_model = deepcopy(proxy_model_init)
        proxy_model = proxy_model.to(DEVICE)
        optimizer = torch.optim.SGD(proxy_model.parameters(), lr=lr, weight_decay=wd, momentum=mom, nesterov=True)
        model = training_loop(epochs, proxy_model, X_train_rem, y_train_rem, criterion, optimizer)
        data_set_rem = CompressedDataset(X_train_rem, y_train_rem)
        mean2, prec2 = get_mean_and_prec(data_set, model, mode=laptype)

        if laptype == "full":
            kl = _computeKL_from_full(mean1, mean2, prec1, prec2)
        elif laptype == "diag":
            kl = _computeKL(mean1, mean2, prec1, prec2)
        elif laptype == "kron":
            kl = _computeKL_from_kron(mean1, mean2, prec1, prec2)
        else:
            raise ValueError("Unknown laptype: {}".format(laptype))
        kl_values.append(kl)
        print(f"KL divergence for index {index}: {kl}")
    # Save the KL divergence values to a file
    output_dir = os.path.join("output", "kl_divergences")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "kl_divergences.pkl")


    return kl_values
