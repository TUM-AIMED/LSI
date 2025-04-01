import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import pickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# Load the IMDB dataset
dataset = load_dataset('imdb')

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(DEVICE)

def encode_review(review, label):
    # Tokenize the review text
    inputs = tokenizer(review, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    # Get the BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the CLS token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].flatten(start_dim=1)
    
    return cls_embedding

# Encode the first 10 reviews as an example (change the number as needed)
# num_reviews = len(dataset['train']['text'])
# encoded_reviews_train = np.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional
# for i, (review, label) in tqdm(enumerate(zip(dataset['train']['text'][:num_reviews], dataset['train']['label'][:num_reviews]))):
#     encoded_reviews_train[i] = encode_review(review, label)

# num_reviews = len(dataset['test']['text'])
# encoded_reviews_test = np.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional
# for i, (review, label) in tqdm(enumerate(zip(dataset['test']['text'][:num_reviews], dataset['test']['label'][:num_reviews]))):
#     encoded_reviews_test[i] = encode_review(review, label)
# batch_size = 200


# num_reviews = len(dataset['train']['text'])
# encoded_reviews_train = torch.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional
# for start_idx in tqdm(range(0, num_reviews, batch_size)):
#     end_idx = min(start_idx + batch_size, num_reviews)
#     batch_reviews = dataset['train']['text'][start_idx:end_idx]
#     encoded_reviews_temp = encode_review(batch_reviews, None)
#     encoded_reviews_train[start_idx:end_idx, :] = encoded_reviews_temp

# num_reviews = len(dataset['test']['text'])
# encoded_reviews_test = torch.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional
# for start_idx in tqdm(range(0, num_reviews, batch_size)):
#     end_idx = min(start_idx + batch_size, num_reviews)
#     batch_reviews = dataset['test']['text'][start_idx:end_idx]
#     encoded_reviews_temp = encode_review(batch_reviews, None)
#     encoded_reviews_test[start_idx:end_idx, :] = encoded_reviews_temp

import random
import lorem

# Generate Lorem Ipsum text
batch_size = 200
lorem_text = [lorem.paragraph() for i in range(5000)]  # Generating 5000 paragraphs for sufficient text
num_reviews = len(lorem_text)
train_labels = torch.squeeze(torch.randint(0, 2, (num_reviews, 1)))
encoded_reviews_train = torch.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional
for start_idx in tqdm(range(0, num_reviews, batch_size)):
    end_idx = min(start_idx + batch_size, num_reviews)
    batch_reviews = lorem_text[start_idx:end_idx]
    encoded_reviews_temp = encode_review(batch_reviews, None)
    encoded_reviews_train[start_idx:end_idx, :] = encoded_reviews_temp

# Save the tensor to a .pt file

base_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_bert_headless/LoremIpsum2"
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Folder created at: {base_path}")
else:
    print(f"Folder already exists at: {base_path}")
torch.save(encoded_reviews_train.clone().detach().cpu(), os.path.join(base_path, 'train_data.pt'))
# torch.save(encoded_reviews_test.clone().detach().cpu(), os.path.join(base_path, 'test_data.pt'))

torch.save(train_labels.cpu(), os.path.join(base_path, 'train_targets.pt'))
# torch.save(test_labels.cpu(), os.path.join(base_path, 'test_targets.pt'))

base_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/LoremIpsum"
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Folder created at: {base_path}")
else:
    print(f"Folder already exists at: {base_path}")
file_name = "lorem_train.pkl"
with open(file_name, "wb") as file:
    # Serialize and save the list to the file
    pickle.dump(lorem_text, file)