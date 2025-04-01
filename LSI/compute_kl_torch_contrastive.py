import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from utils.kl_div import _computeKL
from torch.utils.data import TensorDataset
from laplace import Laplace
from copy import deepcopy
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os
import torch.nn.functional as F
import pickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")


def get_mean_and_prec(img_train, text_train, model, mode = "diag"):

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(img_train[0:8], text_train[0:8]),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    # print(DEVICE)
    if mode == "diag":
        la = Laplace((model[0].to(DEVICE), model[1].to(DEVICE)), 'clip',
                    subset_of_weights='all',
                    hessian_structure='diag')
        la.fit(train_loader)
    elif mode == "full":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights='all',
            hessian_structure='full')
        la.fit(train_loader)
    mean = la.mean
    mean = [m.cpu().numpy() for m in mean]
    post_prec = la.posterior_precision
    post_prec = [p.cpu().numpy() for p in post_prec]
    return mean, post_prec

# Define the dataset class
class CocoDataset(Dataset):
    def __init__(self, coco_cap, coco_ann, image_dir, transform=None, classes=[52, 22], image_count=1000):
        self.coco_cap = coco_cap
        self.coco_ann = coco_ann
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = []
        for img_id in self.coco_ann.imgs:
            anns = self.coco_ann.loadAnns(self.coco_ann.getAnnIds(imgIds=img_id))
            cap = self.coco_cap.loadAnns(self.coco_cap.getAnnIds(imgIds=img_id))[0]['caption']
            if len(cap) < 10:
                print(cap)
            if any(ann['category_id'] in classes for ann in anns):
                self.image_ids.append(img_id)
        self.image_ids = self.image_ids[0:image_count]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann = self.coco_cap.loadAnns(self.coco_cap.getAnnIds(imgIds=image_id))[0]
        caption = ann['caption']
        image_info = self.coco_cap.loadImgs(image_id)[0]
        image_path = self.image_dir + "/" + image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption, idx

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained models
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
resnet.to(DEVICE).eval()  # Set to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.to(DEVICE).eval()  # Set to evaluation mode

# Freeze parameters
for param in resnet.parameters():
    param.requires_grad = False

for param in bert.parameters():
    param.requires_grad = False

# Define projection heads
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Contrastive loss function
class NT_XentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        temperature = self.temperature
        text_embeddings = z_j
        image_embeddings = z_i
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        # Compute the cosine similarity
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t()) / temperature

        # Generate the labels
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=text_embeddings.device)

        # Compute the cross-entropy loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)

        # Total loss is the average of both losses
        total_loss = (loss_img + loss_text) / 2.0

        return total_loss

# Precompute tokenization and embeddings
def precompute_embeddings(folder, resnet, bert, tokenizer):
    if os.path.exists(folder):
        print(f"Loading embeddings from files")
        tensor1_path = os.path.join(folder, 'image_tensor.pt')
        tensor2_path = os.path.join(folder, 'text_tensor.pt')
        tensor3_path = os.path.join(folder, 'ids.pt')

        image_tensor = torch.load(tensor1_path)
        text_tensor = torch.load(tensor2_path)
        id_tensor = torch.load(tensor3_path)
    else:
                # Load the COCO dataset
        coco_cap = COCO('/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/coco/annotations/captions_train2017.json')
        coco_ann = COCO('/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/coco/annotations/instances_train2017.json')
        image_dir = '/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/coco/train2017'
        dataset = CocoDataset(coco_cap, coco_ann, image_dir, transform=transform)

        ids = dataset.image_ids

        image_embeddings = []
        text_embeddings = []
        train_ids = []
        trainloader = DataLoader(dataset, batch_size=64, shuffle=False)
        for image, caption, _ in tqdm(trainloader):
            with torch.no_grad():
                # Image embeddings
                image_embedding = resnet(image.to(DEVICE)).cpu()
                image_embeddings.append(image_embedding.squeeze())

                # Text embeddings
                tokenized_caption = tokenizer(caption, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
                tokenized_caption = {k: v.to(DEVICE) for k, v in tokenized_caption.items()}
                text_embedding = bert(**tokenized_caption).last_hidden_state[:, 0, :].cpu()
                text_embeddings.append(text_embedding)
           
        image_tensor = torch.cat(image_embeddings, dim=0)
        text_tensor = torch.cat(text_embeddings, dim=0)
        id_tensor = torch.tensor(ids)
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Paths to save the tensors
        tensor1_path = os.path.join(folder, 'image_tensor.pt')
        tensor2_path = os.path.join(folder, 'text_tensor.pt')
        tensor3_path = os.path.join(folder, 'ids.pt')

        # Save the tensors
        torch.save(image_tensor, tensor1_path)
        torch.save(text_tensor, tensor2_path)
        torch.save(id_tensor, tensor3_path)

        print(f"Tensor1 saved to {tensor1_path}")
        print(f"Tensor2 saved to {tensor2_path}")
        print(f"Tensor3 saved to {tensor3_path}")
    return image_tensor, text_tensor, id_tensor

# Training function
def train_model(model_idx, model, image_embeddings, text_embeddings, criterion, optimizer, num_epochs=1000):
    model[0].train()
    model[1].train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        image_proj = model[0](image_embeddings)
        text_proj = model[1](text_embeddings)
        loss = criterion(image_proj, text_proj)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}, Model {model_idx + 1}, Loss: {loss.item():.4f}')
    return loss, model

kl = []
idx = []
pred = []
square_diff = []
embedding_dim = 256
# Training process
data_embedded_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_mm_compressed_by_resnet50_and_bert/coco_elephant_banana"
N_SEEDS = 1
image_embeddings, text_embeddings, ids = precompute_embeddings(data_embedded_path, resnet, bert, tokenizer)
for seed in range(N_SEEDS):
    torch.manual_seed(seed + 5)
    n = len(image_embeddings)
    indices = list(range(len(image_embeddings)))
    img_train = deepcopy(image_embeddings[indices]).to(DEVICE)
    text_train = deepcopy(text_embeddings[indices]).to(DEVICE)
    # Initialize projection heads
    image_projection_init = deepcopy(ProjectionHead(2048, embedding_dim).to(DEVICE))
    text_projection_init = deepcopy(ProjectionHead(768, embedding_dim).to(DEVICE))

    image_projection = deepcopy(image_projection_init)
    text_projection = deepcopy(text_projection_init)
    model = (image_projection, text_projection)
    base_model = (image_projection, text_projection)

    # Define loss and optimizer
    criterion = NT_XentLoss().to(DEVICE)
    optimizer = optim.Adam(list(base_model[0].parameters()) + list(base_model[1].parameters()), lr=0.01)

    # Train the model
    train_loss, base_model = train_model(-1, base_model, img_train, text_train, criterion, optimizer)
    print(f'Basemodel trained, Loss: {train_loss:.4f}')
    mean1_1, prec1_1 = get_mean_and_prec(img_train.cpu(), text_train.cpu(), base_model)

    kl_seed = []
    idx_seed = []
    pred_seed = []

    indices_init = list(range(len(image_embeddings)))

    for i in tqdm(range(n)):
        indices = deepcopy(indices_init)
        indices.pop(i)
        img_train = deepcopy(image_embeddings[indices]).to(DEVICE)
        text_train = deepcopy(text_embeddings[indices]).to(DEVICE)
        # Initialize projection heads
        image_projection = deepcopy(image_projection_init)
        text_projection = deepcopy(text_projection_init)
        model = (image_projection, text_projection)

        # Define loss and optimizer
        criterion = NT_XentLoss().to(DEVICE)
        optimizer = optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr=0.01)

        # Train the model
        train_loss, model = train_model(i, model, img_train, text_train, criterion, optimizer)
        print(f'Model {i+1}, Loss: {train_loss:.4f}')
        mean2_1, prec2_1 = get_mean_and_prec(img_train.cpu(), text_train.cpu(), model)
        kl1_1, square_diff1_1 = _computeKL(mean1_1[0], mean2_1[0], prec1_1[0], prec2_1[0])
        kl1_2, square_diff1_2 = _computeKL(mean1_1[1], mean2_1[1], prec1_1[1], prec2_1[1])
        print(f" KL {kl1_1}")
        print(f" KL {kl1_2}")
        kl_seed.append([kl1_1, kl1_2])
        square_diff.append([square_diff1_1, square_diff1_2])
        idx_seed.append(i)
    
    kl.append(kl_seed)
    idx.append(idx_seed)


res = {
    "kl": kl,
    "idx": idx_seed,
    "idx_all": ids
}

filename = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_coco/coco_unsupervised3.pkl'

# Open the file in write-binary mode
with open(filename, 'wb') as file:
    # Use pickle.dump() to write the dictionary to the file
    pickle.dump(res, file)

print(f"Dictionary has been saved as {filename}")
