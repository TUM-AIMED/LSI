import os
import random
import torch
from tqdm import tqdm
from LSI.Datasets.dataset_helper import get_dataset



def get_compressed_path(model, dataset_name, train_or_tests, path):
    short_path = f"/compressed_by_{model}/{dataset_name}/" 
    base_path = path + f"/compressed_by_{model}/{dataset_name}/"
    path_data = path + f"/compressed_by_{model}/{dataset_name}/" + f"{train_or_tests}_data.pt"
    path_target = path + f"/compressed_by_{model}/{dataset_name}/" + f"{train_or_tests}_target.pt"
    return short_path, base_path, path_data, path_target

def compress_dataset(model, dataloader, base_path, path_data, path_target, DEVICE):
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    model.eval()
    targets = []
    idxs = []
    reses = []
    with torch.no_grad():
        for _, (data, target, idx, _) in tqdm(enumerate(dataloader)):
            torch.cuda.empty_cache()
            data, target = data.to(DEVICE), target.to(DEVICE)
            model = model.to(DEVICE)
            # Normalize the data to have values between 0 and 1
            data = data / 255.0
            # Ensure the data has the correct shape for 2D convolutions (N, C, H, W)
            if len(data.shape) == 3:  # If data is (N, H, W), add a channel dimension
                data = data.unsqueeze(1)
            # try:
            res = model.pass_except_last(data)
            # except:
            #     print("Model does not provide a function pass_except_last")
            targets.extend(target)
            idxs.extend(idx)
            reses.extend(torch.squeeze(res).cpu())
            del data
            del target
    targets = [tar.cpu().item() for tar in targets]
    idxs = [id_in.cpu().item() for id_in in idxs]
    torch.save(torch.stack(reses), path_data)
    torch.save(torch.tensor(targets), path_target)
    return