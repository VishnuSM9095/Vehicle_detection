import torch

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            # Process outputs and calculate metrics
