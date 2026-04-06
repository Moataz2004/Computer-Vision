import os
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "D:/Programs/PyCharm/pythonProject/Best Models/Binary_Segmentation.pth"
INPUT_PATH = "D:/FCIS/Computer Vision/Project Data/Fruit/f"
IMG_SIZE = 512

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2(),
])

model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def overlay_image(img_tensor, mask_prob):
    inv_mean = np.array([0.485,0.456,0.406])
    inv_std  = np.array([0.229,0.224,0.225])
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    img = np.clip(img * inv_std + inv_mean, 0, 1)
    mask_color = np.zeros_like(img)
    if mask_prob.ndim == 3: mask_prob = mask_prob.squeeze(0)
    mask_color[...,0] = mask_prob
    overlay = img*0.6 + mask_color*0.4
    return overlay

if os.path.isdir(INPUT_PATH):
    image_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH)
                   if f.lower().endswith((".jpg",".png",".jpeg",".bmp",".tif",".tiff"))]
else:
    image_files = [INPUT_PATH]

with torch.no_grad():
    for img_path in image_files:
        img = np.array(Image.open(img_path).convert("RGB"))
        img = img.astype(np.float32)/255.0   # تأكد من float
        augmented = transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE, dtype=torch.float)
        output = model(img_tensor)['out']
        probs = torch.sigmoid(output).cpu().float().numpy()
        overlay = overlay_image(img_tensor[0].cpu(), probs[0,0])
        plt.figure(figsize=(6,6))
        plt.imshow(overlay)
        plt.title(Path(img_path).name)
        plt.axis('off')
        plt.show()
