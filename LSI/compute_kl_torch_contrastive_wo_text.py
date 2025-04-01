import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from laplace import Laplace
from copy import deepcopy
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Load the COCO dataset
coco = COCO('/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/coco/annotations/instances_train2017.json')
image_dir = '/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/coco/train2017'

# Define the dataset class
class CocoDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None, classes=[3, 20]):
        self.coco = coco
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = []
        for img_id in self.coco.imgs:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            if any(ann['category_id'] in classes for ann in anns):
                self.image_ids.append(img_id)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir + "/" + image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, idx

# Define augmentation transform
augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
full_dataset = CocoDataset(coco, image_dir, transform=transform)

# Load pre-trained model
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
resnet.to(DEVICE).eval()  # Set to evaluation mode

# Freeze parameters
for param in resnet.parameters():
    param.requires_grad = False

# Define projection head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class NT_XentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, labels):
        batch_size = z_i.size(0)
        similarity_matrix = self.cosine_similarity(z_i.unsqueeze(1), z_i.unsqueeze(0)) / self.temperature

        # Create labels for positive pairs
        labels = labels.to(DEVICE)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(DEVICE)
        
        # Mask out self-comparisons
        mask.fill_diagonal_(0)
        
        # Compute loss
        positive_samples = similarity_matrix * mask
        negative_samples = similarity_matrix * (1 - mask)
        
        # Compute NT-Xent loss
        loss = -torch.log(positive_samples.sum(dim=-1) / (positive_samples.sum(dim=-1) + negative_samples.sum(dim=-1)))
        return loss.mean()


# Precompute image embeddings and create augmented pairs
def precompute_image_embeddings(dataset, augmentation_transform, folder, resnet):
    if os.path.exists(folder):
        print(f"Loading embeddings from files")
        tensor1_path = os.path.join(folder, 'image_tensor.pt')
        tensor2_path = os.path.join(folder, 'idxs_tensor.pt')

        image_tensor = torch.load(tensor1_path)
        idxs_tensor = torch.load(tensor2_path)
    else:
        og_images = []
        image_embeddings = []
        idxs = []
        trainloader = DataLoader(dataset, batch_size=64, shuffle=True)
        for image, idx in tqdm(trainloader):
            with torch.no_grad():
                # Original Image embeddings
                image_embedding = resnet(image.to(DEVICE)).cpu()
                image_embeddings.append(image_embedding.squeeze())
                og_images.append(image.cpu())
                idxs.append(idx)

                # Augmented Image embeddings
                for j in range(5):
                    augmented_images = augmentation_transform(image)
                    image_embedding = resnet(augmented_images.to(DEVICE)).cpu()
                    image_embeddings.append(image_embedding.squeeze())
                    og_images.append(augmented_images.cpu())
                    idxs.append(idx)
           
        image_tensor = torch.cat(image_embeddings, dim=0)
        idxs_tensor = torch.cat(idxs, dim=0)
        og_images_tensor = torch.cat(og_images, dim=0)
        
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Paths to save the tensors
        tensor1_path = os.path.join(folder, 'image_tensor.pt')
        tensor2_path = os.path.join(folder, 'idxs_tensor.pt')
        tensor3_path = os.path.join(folder, 'og_image_tensor.pt')

        # Save the tensors
        torch.save(image_tensor, tensor1_path)
        torch.save(idxs_tensor, tensor2_path)
        torch.save(og_images_tensor, tensor3_path)


        print(f"Tensor1 saved to {tensor1_path}")
        print(f"Tensor2 saved to {tensor2_path}")
    return image_tensor, idxs_tensor

# Training function
def train_model(model_idx, model, image_embeddings, idxs, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        image_proj = model(image_embeddings)
        loss = criterion(image_proj, idxs)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Model {model_idx + 1}, Loss: {loss.item():.4f}')
    return model

kl = []
idx = []
pred = []
square_diff = []
# Training process
data_embedded_path = "/vol/aimspace/users/kaiserj/Datasets/Datasets_mm_compressed_by_resnet50_and_bert/coco3"
N_SEEDS = 1
image_embeddings, idxs = precompute_image_embeddings(full_dataset, augmentation_transform, data_embedded_path, resnet)
for seed in range(N_SEEDS):
    torch.manual_seed(seed + 5)
    n = len(full_dataset)
    
    image_embeddings = deepcopy(image_embeddings).to(DEVICE)
    idxs = deepcopy(idxs).to(DEVICE)
    # Initialize projection head
    image_projection = ProjectionHead(2048, 256).to(DEVICE)

    # Define loss and optimizer
    criterion = NT_XentLoss().to(DEVICE)
    optimizer = optim.Adam(image_projection.parameters(), lr=0.001)

    # Train the model
    train_loss = train_model(-1, image_projection, image_embeddings, idxs, criterion, optimizer, num_epochs=10)
    print(f'Basemodel trained, Loss: {train_loss:.4f}')
    mean1_1, prec1_1 = get_mean_and_prec(img_train1.cpu(), np.zeros(img_train1.size(0)), image_projection)

    kl_seed = []
    idx_seed = []
    pred_seed = []
    # TODO not include image and all its augmentations
    for i in range(n):
        indices = list(range(len(image_embeddings)))
        indices.pop(i)
        img_train1 = deepcopy(image_embeddings[indices]).to(DEVICE)
        img_train2 = deepcopy(image_embeddings[indices]).to(DEVICE)
        # Initialize projection head
        image_projection = ProjectionHead(2048, 256).to(DEVICE)

        # Define loss and optimizer
        criterion = NT_XentLoss().to(DEVICE)
        optimizer = optim.Adam(image_projection.parameters(), lr=0.001)

        # Train the model
       
