import torch


def get_dimensions(img):
    # torchvision has some backward compatibility issues with torchvision.transforms.functional.get_dimensions
    if torch.is_tensor(img):
        c, h, w = img.shape
    else:
        w, h = img.size
        c = 1 if img.mode == "L" else 3
    return c, h, w
