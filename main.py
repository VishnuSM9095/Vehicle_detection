import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from preprocess.load_data import load_images
from preprocess.preprocess_data import preprocess_images
from model.model import get_model
from model.train import train_model
from evaluate.evaluate import evaluate_model
from deploy.detect import load_trained_model, detect_objects

class CustomDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.annotations[idx]
        image = F.to_tensor(image)
        return image, annotation

# Set directories
image_dir = 'data/images/'
annotations_dir = 'data/annotations/'

# Load and preprocess data
images = load_images(image_dir)
preprocessed_images = preprocess_images(images)

# Create dummy annotations (replace this with your actual annotations)
# Assuming annotations is a list of dictionaries as required by torchvision
annotations = [{'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64)} for _ in preprocessed_images]

# Prepare DataLoader, Model, Optimizer
dataset = CustomDataset(preprocessed_images, annotations)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_classes = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, optimizer, num_epochs=10, device=device)

# Evaluate the model (use a validation dataset)
# evaluate_model(model, validation_dataloader, device=device)

# Save the model
torch.save(model.state_dict(), 'model_weights.pth')

# Load and use the model for detection
trained_model = load_trained_model('model_weights.pth', num_classes, device)
image = F.to_tensor(preprocessed_images[0]).unsqueeze(0).to(device)  # Example usage
predictions = detect_objects(trained_model, image, device)
print(predictions)
