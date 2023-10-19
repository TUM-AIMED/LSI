import pickle
import matplotlib.pyplot as plt
import numpy as np

path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_feld/feldman_cifar10_5.pkl"
with open(path, 'rb') as f:
    data = pickle.load(f) 
trainset_correctness = data["trainset_correct"]
trainset_mask = data["trainset_mask"]
inv_mask = data["inv_mask"]
def _masked_avg(x, mask, axis=0, esp=1e-10):
    test1 = np.sum(x * mask, axis=axis)
    test2 = np.maximum(np.sum(mask, axis=axis), esp)
    return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

trainset_correctness2=trainset_correctness[:, 0:10]
trainset_mask2=trainset_mask[:, 0:10]
inv_mask2=inv_mask[:, 0:10]
print(trainset_correctness2)
print(trainset_mask2)
print(inv_mask2)

mem1 = _masked_avg(trainset_correctness, trainset_mask)
mem2 = _masked_avg(trainset_correctness, inv_mask)
mem_est = mem1 - mem2

plt.hist(mem_est, bins=30, color='blue', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.savefig('histogram.png')

print("")
