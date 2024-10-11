import torch

def load_trained_model(weights_path, num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model

def detect_objects(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        predictions = model([image])
    return predictions
