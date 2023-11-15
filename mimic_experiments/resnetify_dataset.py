import sys
import os

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset

import torch
import numpy as np
import warnings
import json
import argparse
import time

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--model", type=str, default="resnet18_headless", help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Value for lerining_rate (optional)")


    args = parser.parse_args()

    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    params = {}
    params["model"] = {}
    params["model"]["model"] = args.model
    params["model"]["dataset_name"] = args.dataset
    params["save_path"] = "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_" + params["model"]["model"] + "/" + params["model"]["dataset_name"]

    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])
 
    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=len(data_set), # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=len(data_set_test), # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
        model = model_class(len(torch.flatten(train_X_example))).to(DEVICE)
    else:
        model = model_class()
        model = model.to(DEVICE)

    def compress_dataset(model, dataset):
        model.eval()
        start_time = time.time()
        for _, (data, target, idx, _) in enumerate(dataset):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
        return torch.squeeze(output), target

    data, target = compress_dataset(model, train_loader)
    data_test, target_test = compress_dataset(model, test_loader)
    if not os.path.exists(params["save_path"]):
        os.makedirs(params["save_path"])
    torch.save(data, params["save_path"] + "/train_data.pt")
    torch.save(target, params["save_path"] + "/train_target.pt")
    torch.save(data_test, params["save_path"] + "/test_data.pt")
    torch.save(target_test, params["save_path"] + "/test_target.pt")
    