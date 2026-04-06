import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "D:/Programs/PyCharm/pythonProject/Best Models/Multi_Class_Segmentation.pth"
TEST_DIR   = "D:/FCIS/Computer Vision/Project Data/Fruit/f"

class UNetResNet34(nn.Module):
    def __init__(self, n_classes=31, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34
        self.encoder = resnet34(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.enc1 = nn.Sequential(*self.encoder_layers[:3])
        self.enc2 = nn.Sequential(*self.encoder_layers[3:5])
        self.enc3 = self.encoder_layers[5]
        self.enc4 = self.encoder_layers[6]
        self.enc5 = self.encoder_layers[7]

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = nn.Conv2d(512, 256, 3, padding=1)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = nn.Conv2d(256, 128, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Conv2d(32, 32, 3, padding=1)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.up5(e5)
        d5 = self.dec5(torch.cat([d5, e4], dim=1))
        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


num_classes = 31

model = UNetResNet34(n_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def preprocess_image(img_path, img_size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = T.ToTensor()(img_resized).unsqueeze(0)  # batch
    return img_tensor.to(device), img_resized

def predict_image(img_path):
    img_tensor, img_resized = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    return img_resized, mask

def visualize_mask(img, mask):
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    mask_color = colors[mask]
    overlay = cv2.addWeighted(img, 0.5, mask_color, 0.5, 0)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Segmentation Mask")
    plt.imshow(mask_color)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img, mask = predict_image(img_path)
            visualize_mask(img, mask)



process_folder(TEST_DIR)
