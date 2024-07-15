import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np

# Load the IMDB dataset
dataset = load_dataset('imdb')

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def encode_review(review):
    # Tokenize the review text
    inputs = tokenizer(review, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Get the BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the CLS token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    return cls_embedding

# Encode the first 10 reviews as an example (change the number as needed)
num_reviews = 10
encoded_reviews = np.zeros((num_reviews, 768))  # BERT's CLS embedding is 768-dimensional

for i, review in enumerate(dataset['train']['text'][:num_reviews]):
    encoded_reviews[i] = encode_review(review)

# Print the shape of the encoded reviews
print(encoded_reviews.shape)

# Save the encoded reviews to a file
np.save('encoded_imdb_reviews.npy', encoded_reviews)
