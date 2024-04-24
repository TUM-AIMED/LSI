import sys
import os

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from tqdm import tqdm
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
    parser.add_argument("--dataset", type=str, default="Prima", help="Value for lerining_rate (optional)")


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
        batch_size=1, # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=1, # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
        train_X_example = 10
        model = model_class(len(torch.flatten(train_X_example))).cuda()
    else:
        model = model_class(pretrained="Imagenet")
        model = model.cuda()

    def compress_dataset(model, dataset, path):
        if not os.path.exists(params["save_path"]):
            os.makedirs(params["save_path"])

        model.eval()
        targets = []
        idxs = []
        reses = []
        with torch.no_grad():
            for _, (data, target, idx, _) in tqdm(enumerate(dataset)):
                torch.cuda.empty_cache()
                data, target = data.cuda(), target.cuda()
                res = model(data)
                targets.append(target.cpu().item())
                idxs.append(idx.item())
                reses.append(torch.squeeze(res).cpu())
                del data
                del target
        torch.save(reses, params["save_path"] + "/" + path + "_data.pt")
        torch.save(targets, params["save_path"] + "/" + path + "_target.pt")

        print(f'saving under {params["save_path"] + "/"}')
        return 
    
    # compress_dataset(model, test_loader, "train")
    compress_dataset(model, train_loader, "train")
    compress_dataset(model, test_loader, "test")

    
    