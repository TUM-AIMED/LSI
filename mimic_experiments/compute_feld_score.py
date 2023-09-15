from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
import numpy as np
import random
import torch
import os
import pickle
from copy import deepcopy 
from tqdm import tqdm
import time

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_results(dataset_train_0, output_in, output_out, model_num):
    print("Evaluating results", flush=True)
    results = {}
    count = 0
    for idx in range(0, len(dataset_train_0.data)):
        _, label, _, _ = dataset_train_0[idx]
        in_list = np.array(output_in[idx])
        out_list = np.array(output_out[idx])
        if len(in_list.shape) != 2 or len(out_list.shape) != 2:
            # print(f"Index {idx} was not sampled with len_in {in_list.shape}, len_out {out_list.shape}")
            count += 1
            continue
        # reduce to important class
        min_length = min(in_list.shape[0], out_list.shape[0])
        in_list = in_list[0:min_length, label]
        out_list = out_list[0:min_length, label]
        results[idx] = np.average(in_list - out_list)
    print(f"{count} items were skipped", flush=True)

    final = {}
    final["data"] = results
    final["model_count"] = model_num
    with open(os.path.join(results_path), 'wb') as file:
            pickle.dump(final, file)
    print(f"Save results at model number {model_num} to {results_path}", flush=True)


print("Starting", flush=True)
num_epochs = 150
batchsize = 500
num_models = 1000
save_every = 100

dataset_name ="cifar10"
model_name ="mlp"
criterion = torch.nn.CrossEntropyLoss()
results_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feldman/" + model_name + "_" + dataset_name + "_" + str(num_epochs) + "_" + str(num_models) +".pkl"

dataset_class, dataset_path = get_dataset(dataset_name)
model_class = get_model(model_name)
dataset_train_0 = dataset_class(dataset_path)
train_loader_0 =  torch.utils.data.DataLoader(
        dataset_train_0,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

train_X_example, _, _, _ = train_loader_0.dataset[0]
N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

output_in = {i: [] for i in range(0, len(dataset_train_0.data))}
output_out = {i: [] for i in range(0, len(dataset_train_0.data))}

print("Started Training")
for i in range(num_models):
    start_time = time.time()
    model = model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.0005,
    )
    keep_idx = random.sample(range(0, len(dataset_train_0.data)), int(0.7 * len(dataset_train_0.data)))
    dataset_active = deepcopy(dataset_train_0)
    dataset_active.reduce_to_active(keep_idx)
    train_loader_active =  torch.utils.data.DataLoader(
        dataset_active,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model.train()
    for epoch in range(num_epochs):
        for _, (data, target, idx, _) in enumerate(train_loader_active):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    train_end_time = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (data, target, idx, _) in enumerate(train_loader_0):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            idx = np.array(idx)
            _, predicted = torch.max(outputs.data, 1)
            outputs = outputs.cpu().numpy()

            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            in_mask = np.isin(idx, np.array(keep_idx))
            in_idx = idx[in_mask]
            in_outputs = outputs[in_mask]
            
            out_mask = ~in_mask
            out_idx = idx[out_mask]
            out_outputs = outputs[out_mask]

            for idx, output in zip(in_idx, in_outputs):
                output_in[idx].append(output)
            for idx, output in zip(out_idx, out_outputs):
                output_out[idx].append(output)
    accuracy = correct / total
    print(f"Accuracy of model {i} on all data: {accuracy:.4f} and took {(time.time() - start_time):.4f} seconds, training taking {(train_end_time - start_time):.4f} ", flush=True)
    if i%save_every == 0:
        save_results(dataset_train_0, output_in, output_out, i)
save_results(dataset_train_0, output_in, output_out, num_models)


    







    

