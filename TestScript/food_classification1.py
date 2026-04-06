import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDER = "D:/FCIS/Computer Vision/Project Data/Fruit/f"
MODEL_PATH = r"D:/Programs/PyCharm/pythonProject/Best Models/Food_Classification.pth"

GALLERY_EMB = "D:/Programs/PyCharm/pythonProject/GalleryEmbeddings/gallery_embeddings.npy"
GALLERY_LAB = "D:/Programs/PyCharm/pythonProject/GalleryEmbeddings/gallery_labels.npy"
CLASS_NAMES = "D:/Programs/PyCharm/pythonProject/GalleryEmbeddings/gallery_class_names.txt"

OUTPUT_FILE = "PartB_Output.txt"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=1)

model = EmbeddingNet().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

gallery_emb = np.load(GALLERY_EMB)
gallery_lab = np.load(GALLERY_LAB)

with open(CLASS_NAMES) as f:
    class_names = [l.strip() for l in f.readlines()]

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(gallery_emb, gallery_lab)

results = []

with torch.no_grad():
    for f in os.listdir(INPUT_FOLDER):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            img = Image.open(os.path.join(INPUT_FOLDER, f)).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            emb = model(img).cpu().numpy()
            pred = knn.predict(emb)[0]
            results.append(f"{f} -> {class_names[pred]}")

with open(OUTPUT_FILE, "w") as f:
    for r in results:
        f.write(r + "\n")

print(f"Finished. Results saved to {OUTPUT_FILE}")
