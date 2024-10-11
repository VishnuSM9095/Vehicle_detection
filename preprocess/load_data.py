import cv2
import os

def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is not None:
                images.append(img)
    return images
