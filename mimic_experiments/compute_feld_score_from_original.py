"""Simple Influence / Memorization estimation on MNIST."""
import os
import itertools
import pickle
import numpy as np
import numpy.random as npr
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, test_set):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_set:
            inputs, target, _, _ = data
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy

def batch_correctness(model, data_set):
    length = data_set.data.shape[0]
    train_loader =  torch.utils.data.DataLoader(
        data_set,
        batch_size=data_set.data.shape[0],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )   
    for _, (data, targets, idx, _) in enumerate(train_loader):
        output = model(data)
        _, predicted_class = torch.max(output.data, 1)
        predicted_class = predicted_class.cpu()
        print(f"accuracy on whole data = {sum(predicted_class == targets)/predicted_class.shape[0]}")

        return predicted_class == targets



def subset_train(seed, subset_ratio):
    # TODO insert random seed
    seed = torch.randint(0, 20000, [1,1])
    seed = seed.item()
    torch.manual_seed(42)
    num_epochs = 25
    batch_size = 256
    dataset_name = "cifar10"
    model_name ="resnet18"

    dataset_class, dataset_path = get_dataset(dataset_name)
    model_class = get_model(model_name)
    dataset_train_0 = dataset_class(dataset_path)
    dataset_test_0 = dataset_class(dataset_path, train=False)
    dataset_train = dataset_class(dataset_path)
    criterion = torch.nn.CrossEntropyLoss()

    train_in_len = dataset_train_0.data[0].flatten().shape[0]
    N_CLASSES =  len(np.unique(dataset_train_0.labels))
    model = model_class(train_in_len, N_CLASSES).to(DEVICE)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,
        weight_decay=0.005,
    )

    subset_idx = np.random.choice(len(dataset_train.data), size=int(len(dataset_train.data) * subset_ratio), replace=False)
    dataset_train.reduce_to_active(subset_idx)
    
    train_loader =  torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )    
    test_loader = torch.utils.data.DataLoader(
        dataset_test_0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )   

    train_loader_0 = torch.utils.data.DataLoader(
        dataset_train_0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )  

    for epoch in range(num_epochs):
        model.train()
        total = 0
        correct = 0
        for _, (data, target, idx, _) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = correct/total
        torch.cuda.empty_cache()
        print(f"training accuracy {accuracy}")
        if epoch % 20 == 0:
            print(f"testing accuracy {test(model, test_loader)}")
            print(f"training accuracy - subset - after {test(model, train_loader)}")
            print(f"training accuracy - whole data {test(model, train_loader_0)}")
        optimizer.zero_grad()

    trainset_correctness = batch_correctness(
        model, dataset_train_0)
    testset_correctness = batch_correctness(
        model, dataset_test_0)

    trainset_mask = np.zeros(dataset_train_0.data.shape[0], dtype=np.bool_)
    trainset_mask[subset_idx] = True
    return trainset_mask, np.asarray(trainset_correctness), np.asarray(testset_correctness)


def estimate_infl_mem(n_runs):
  subset_ratio = 0.7
  
  results = []

  for i_run in tqdm(range(n_runs), desc=f'SS Ratio={subset_ratio:.2f}'):
    results.append(subset_train(i_run, subset_ratio))

  trainset_mask = np.vstack([ret[0] for ret in results])
  inv_mask = np.logical_not(trainset_mask)
  trainset_correctness = np.vstack([ret[1] for ret in results])
  testset_correctness = np.vstack([ret[2] for ret in results])

  print(f'Avg test acc = {np.mean(testset_correctness):.4f}')

  def _masked_avg(x, mask, axis=0, esp=1e-10):
    test1 = np.sum(x * mask, axis=axis)
    test2 = np.maximum(np.sum(mask, axis=axis), esp)
    return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

  def _masked_dot(x, mask, esp=1e-10):
    x = x.T.astype(np.float32)
    return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

  mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
  infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)

  return dict(memorization=mem_est, 
              influence=infl_est, 
              trainset_correct=trainset_correctness, 
              trainset_mask=trainset_mask,
              testset_correct=testset_correctness, 
              inv_mask=inv_mask)
  
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_runs", type=int, default=2, help="Value for lerining_rate (optional)")
    args = parser.parse_args()
    results_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld"
    n_runs = args.n_runs
    print(n_runs, flush=True)

    estimates = estimate_infl_mem(n_runs)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(results_path + "/" + "feldman_cifar10_final_" + str(n_runs) + ".pkl", 'wb') as file:
        pickle.dump(estimates, file)

