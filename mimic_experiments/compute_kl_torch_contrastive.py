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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# Load the COCO dataset
coco = COCO('annotations/captions_train2017.json')
image_dir = 'train2017/'


def get_mean_and_prec(data, labels, model, mode = "diag"):
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels).to(torch.long).to(DEVICE)
    data = np.asarray(data)
    data = torch.from_numpy(data).to(torch.float32).to(DEVICE)
    
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    # print(DEVICE)
    if mode == "diag":
        la = Laplace(model.features.to(DEVICE), 'classification',
                    subset_of_weights='all',
                    hessian_structure='diag')
        la.fit(train_loader)
    elif mode == "full":
        la = Laplace(model.features.to(DEVICE), 'classification',
            subset_of_weights='all',
            hessian_structure='full')
        la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec

# Define the dataset class
class CocoDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgToAnns.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))[0]
        caption = ann['caption']
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir + image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
full_dataset = CocoDataset(coco, image_dir, transform=transform)

# Load pre-trained models
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
resnet.eval()  # Set to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()  # Set to evaluation mode

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
        batch_size = z_i.size(0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

# Precompute tokenization and embeddings
def precompute_embeddings(dataset, resnet, bert, tokenizer):
    image_embeddings = []
    text_embeddings = []
    for image, caption, image_id in dataset:
        with torch.no_grad():
            # Image embeddings
            image = image.unsqueeze(0).cuda()
            image_embedding = resnet(image).view(-1).cpu()
            image_embeddings.append(image_embedding)

            # Text embeddings
            tokenized_caption = tokenizer(caption, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
            tokenized_caption = {k: v.cuda() for k, v in tokenized_caption.items()}
            text_embedding = bert(**tokenized_caption).last_hidden_state[:, 0, :].view(-1).cpu()
            text_embeddings.append(text_embedding)
    
    return torch.stack(image_embeddings), torch.stack(text_embeddings)

# Training function
def train_model(model_idx, model, image_embeddings, text_embeddings, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        image_proj = model[0](image_embeddings)
        text_proj = model[1](text_embeddings)
        loss = criterion(image_proj, text_proj)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Model {model_idx + 1}, Loss: {loss.item():.4f}')
    return model

kl = []
idx = []
pred = []
square_diff = []
# Training process
N_SEEDS = 1
image_embeddings, text_embeddings = precompute_embeddings(full_dataset, resnet, bert, tokenizer)
for seed in range(N_SEEDS):
    torch.manual_seed(seed + 5)
    n = len(full_dataset)
    
    img_train = deepcopy(image_embeddings).cuda()
    text_train = deepcopy(text_embeddings).cuda()
    # Initialize projection heads
    image_projection = ProjectionHead(2048, 256).cuda()
    text_projection = ProjectionHead(768, 256).cuda()
    base_model = (image_projection, text_projection)

    # Define loss and optimizer
    criterion = NT_XentLoss().cuda()
    optimizer = optim.Adam(list(base_model[0].parameters()) + list(base_model[1].parameters()), lr=0.001)

    # Train the model
    train_loss = train_model(i, model, img_train, text_train, criterion, optimizer, num_epochs=10)
    print(f'Basemodel trained, Loss: {train_loss:.4f}')
    mean1_1, prec1_1 = get_mean_and_prec(img_train.cpu(), text_train.cpu(), base_model)

    kl_seed = []
    idx_seed = []
    pred_seed = []

    for i in range(n):

        indices = list(range(len(image_embeddings)))
        indices.pop(i)
        img_train = deepcopy(image_embeddings[indices]).cuda()
        text_train = deepcopy(text_embeddings[indices]).cuda()
        # Initialize projection heads
        image_projection = ProjectionHead(2048, 256).cuda()
        text_projection = ProjectionHead(768, 256).cuda()
        model = (image_projection, text_projection)

        # Define loss and optimizer
        criterion = NT_XentLoss().cuda()
        optimizer = optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr=0.001)

        # Train the model
        train_loss = train_model(i, model, img_train, text_train, criterion, optimizer, num_epochs=10)
        print(f'Model {i+1}, Loss: {train_loss:.4f}')
        mean2_1, prec2_1 = get_mean_and_prec(img_train.cpu(), text_train.cpu(), model)
        kl1_1, square_diff1_1 = _computeKL(mean1_1, mean2_1, prec1_1, prec2_1)

        kl_seed.append(kl1_1)
        square_diff.append(square_diff1_1)
        idx_seed.append(i)
    
    kl.append(kl_seed)
    idx.append(idx_seed)


