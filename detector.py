import os
import cv2
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations, transform=None):
        self.images_dir = images_dir
        self.annotations = annotations  # List of annotations
        self.transform = transform
        self.image_files = os.listdir(images_dir)  # List of image files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)

        # Fetch corresponding annotation using the image_id (file name)
        image_id = self.image_files[idx]
        annotation = next((item for item in self.annotations if item['image_id'] == image_id), None)

        if annotation is None:
            # Create a dummy target if no annotations are found
            return image, {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}

        # Convert annotation to the format required by FasterRCNN
        boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)  # Bounding boxes
        labels = torch.tensor(annotation['labels'], dtype=torch.int64)  # Class labels
        target = {"boxes": boxes, "labels": labels}

        return image, target

# Load the trained model
def load_trained_model(model_path, num_classes):
    # Load the pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (num_classes = number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the model state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model

# Object detection function
def detect_objects(image, model, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension
        predictions = model(image_tensor)

    # Process the predictions to count the number of detections
    num_detections = len(predictions[0]['boxes'])  # Get number of detected boxes
    print(f"Detected {num_detections} objects.")

    return predictions

# Main function to run the training and detection
def main():
    # Set parameters
    images_dir = 'path/to/your/images'  # Replace with your image directory
    annotations = [
        # Example annotations format
        # {'image_id': 'image1.png', 'boxes': [[x1, y1, x2, y2]], 'labels': [1]},
        # {'image_id': 'image2.png', 'boxes': [[x1, y1, x2, y2]], 'labels': [2]},
    ]
    num_classes = 91  # Adjust according to your dataset
    model_path = 'path/to/your/model.pth'  # Replace with your model path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the dataset
    dataset = CustomDataset(images_dir, annotations)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load the model
    model = load_trained_model(model_path, num_classes)
    model.to(device)

    # Training loop (placeholder for training logic)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation (placeholder, add optimizer step)
            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item()}')

    # Test the model on a sample image
    sample_image_path = 'sample.png'  # Replace with a sample image path
    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    # Run detection
    detect_objects(sample_image, model, device)

if __name__ == "__main__":
    main()
