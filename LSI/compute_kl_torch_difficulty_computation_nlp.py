import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from transformers import TrainerCallback, TrainingArguments, Trainer



import time
import numpy as np
import torch
import pickle
from copy import deepcopy
from collections import defaultdict
import random
from tqdm import tqdm
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
portions = 3
paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_25000_dataset_Imdbcompressed_model_Tinymodel_subset_25000_range_0_25000_corrupt_0.0_corrupt_data_0.0_0_torch.pkl"]

path_name = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_torch_difficulty_computation_after_workshop"
file_name = "IMDB_DIFFICULTY_COMPUTATION_Trial"
def get_ordering(paths, kind, ytr=None, len_dataset=50000):
    if kind == "random":
        return_list = random.sample(range(len_dataset), len_dataset)
        return return_list
    idxs = []
    kls = []
    for batch, path in enumerate(paths):
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
            kl_data = np.array(final_dict["kl"])
            kl_data = np.mean(kl_data, axis=0)
            idx = list(range(len(kl_data)))
            idx = [idx_i + 10000 * batch for idx_i in idx]
            kls.extend(kl_data)
            idxs.extend(idx)
    while len(kls) < len_dataset:
        kls.append(kls[-1])
        idxs.append(idxs[-1] + 1)
    sorted_data = sorted(zip(kls, idxs), key=lambda x: x[0])
    kl_data, idx = zip(*sorted_data)
    idx = list(idx)
    if kind == "smallest":
        return idx
    elif kind == "largest":
        idx.reverse()
        return idx
    elif kind== "largest_balanced":
        idx.reverse()
        classes_ordered = []
        for i in torch.unique(ytr):
            class_i_idx = [index for index, val in enumerate(ytr) if val == i]
            class_i_order = [index for index in idx if index in class_i_idx]
            classes_ordered.append(class_i_order)
        initial_lengths = [len(class_i_order) for class_i_order in classes_ordered]
        
        idx_new = []
        while any([len(class_i_order)>0 for class_i_order in classes_ordered]):
            current_quotients = [len(class_i_order)/ initial_length for class_i_order, initial_length in zip(classes_ordered, initial_lengths)]
            current_largest = current_quotients.index(max(current_quotients))
            idx_new.append(classes_ordered[current_largest].pop(0))

        # idx_new = list(chain(*zip(*classes_ordered)))
        return idx_new
    else:
        raise Exception("Not implemented")
    return


dataset = load_dataset('imdb')
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset_init = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

y_train = train_dataset_init["labels"]
y_test = eval_dataset["labels"]

train_unique = y_train.unique(return_counts=True)
test_unique = y_test.unique(return_counts=True)

train_unique_acc = torch.max(train_unique[1])/torch.sum(train_unique[1])
test_unique_acc = torch.max(test_unique[1])/torch.sum(test_unique[1])


ordering_largest_balanced = get_ordering(paths, "largest_balanced", ytr=y_train, len_dataset=25000)
orderings = {
    "ordering_largest_balanced": ordering_largest_balanced,
}

results_performance_train = defaultdict(dict)
results_performance_test = defaultdict(dict)

torch.manual_seed(5)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
backup_model = deepcopy(model)

for ordering_name, order in orderings.items():
        for i in tqdm(range(0, portions)):
            model = deepcopy(backup_model).to(DEVICE)
            portion = len(train_dataset_init) * 1/(portions)
            subset_idx = order[int(i*portion):int((i+1)*portion)]
            train_dataset = deepcopy(train_dataset_init.select([i for i in subset_idx]))

            start_time = time.time()

            training_args = TrainingArguments(
                output_dir="./results_nlp_difficulty",
                eval_strategy="steps",
                eval_steps = 50,
                learning_rate=2e-5,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                num_train_epochs=7,
                weight_decay=0.01,
            )

            metric = load_metric("accuracy")

            def compute_metrics(p):
                predictions, labels = p
                preds = predictions.argmax(axis=1)
                return metric.compute(predictions=preds, references=labels)

            train_accuracies = []
            test_accuracies = []
            class PrinterCallback(TrainerCallback):
                def on_epoch_end(self, args, state, control, **kwargs):
                    # Evaluate on train dataset
                    train_result = trainer.evaluate(eval_dataset=train_dataset)
                    train_accuracies.append(train_result['eval_accuracy'])
                    print(f"Train Accuracy after epoch {state.epoch}: {train_result['eval_accuracy']:.4f}")

                    # Evaluate on eval dataset
                    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                    test_accuracies.append(eval_result['eval_accuracy'])
                    print(f"Test Accuracy after epoch {state.epoch}: {eval_result['eval_accuracy']:.4f}")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                callbacks=[PrinterCallback]
            )

            trainer.train()


            results_performance_train[ordering_name][i] = train_accuracies
            results_performance_test[ordering_name][i] = test_accuracies

result = {"train_acc_subset": results_performance_train,
            "test_acc": results_performance_test,
            "train_unique": train_unique_acc,
            "test_unique": test_unique_acc
            }


if not os.path.exists(path_name):
    os.makedirs(path_name)
file_path = os.path.join(path_name, file_name + ".pkl")
with open(file_path, 'wb') as file:
    pickle.dump(result, file)
print(f'Saving at {file_path}')

