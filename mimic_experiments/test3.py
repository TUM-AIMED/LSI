import os
import torch
from tqdm import tqdm


data_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Imagenet/train_data"
save_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Imagenet_chunks/train_data"
if not os.path.exists(save_path):
    os.makedirs(save_path)
train_data_paths = os.listdir(data_path)
train_data = []
idx_path = []
for i, indiv_path in tqdm(enumerate(train_data_paths, start=1)):
    with open(os.path.join(data_path, indiv_path), 'rb') as fo:
        train_data.append(torch.load(fo))
        idx_path.append(indiv_path)
    if i % 10000 == 0:
        torch.save([train_data, idx_path], save_path + str(i.item()) + ".pt")
        print(f'Saved as {save_path + str(i.item()) + ".pt"}')