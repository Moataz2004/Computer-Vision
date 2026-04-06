import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOLDER_PATH = "D:/FCIS/Computer Vision/Project Data/Fruit/f/TestCases/Siamese Case II Test"
MODEL_PATH = r"D:/Programs/PyCharm/pythonProject/Best Models/Food_Classification.pth"

THRESHOLD = 0.5

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

embeddings = {}
anchor_emb = None

with torch.no_grad():
    for f in os.listdir(FOLDER_PATH):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            img = Image.open(os.path.join(FOLDER_PATH, f)).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            emb = model(img).cpu().numpy()

            if f.lower().startswith("anchor"):
                anchor_emb = emb
            else:
                class_name = f.split("_")[0]
                embeddings.setdefault(class_name, []).append(emb)

if anchor_emb is None:
    raise ValueError("Anchor image not found")

best_class = None
best_dist = 1e9

for cls, embs in embeddings.items():
    embs = np.vstack(embs)
    sims = cosine_similarity(anchor_emb, embs)
    dist = 1 - sims.max()

    if dist < best_dist:
        best_dist = dist
        best_class = cls

print("Best Match:", best_class)
print("Distance:", round(best_dist, 4))

if best_dist < THRESHOLD:
    print(f"Anchor belongs to class: {best_class}")
else:
    print("No Match")
