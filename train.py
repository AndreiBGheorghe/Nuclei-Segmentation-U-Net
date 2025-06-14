import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from data_loader import NucleiDataset
from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = NucleiDataset("dataset", transform=ToTensor(), mask_transform=ToTensor())
# dataset = torch.utils.data.Subset(full_dataset, range(50))
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = UNet().to(device)
if os.path.exists("unet.pth"):
    model.load_state_dict(torch.load("unet.pth", map_location=device))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"{epoch+1} - {epoch_loss:.4f}")

torch.save(model.state_dict(), "unet.pth")