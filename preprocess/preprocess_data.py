import numpy as np
from imgaug import augmenters as iaa

def preprocess_images(images):
    seq = iaa.Sequential([
        iaa.Resize({"height": 416, "width": 416}),
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Multiply((0.8, 1.2)),
    ])
    return seq(images=images)
