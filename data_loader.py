import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class NucleiDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        for folder in os.listdir(root_dir):
            img_path = os.path.join(root_dir, folder, "images", os.listdir(os.path.join(root_dir, folder, "images"))[0])
            mask_folder = os.path.join(root_dir, folder, "masks")
            mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder)]
            self.samples.append((img_path, mask_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_paths = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = np.zeros((256, 256), dtype=np.uint8)
        for m_path in mask_paths:
            m = Image.open(m_path).resize((256, 256)).convert("L")
            m_bin = (np.array(m) > 0).astype(np.uint8)
            mask = np.maximum(mask, m_bin)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(Image.fromarray(mask))
        else:
            mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask