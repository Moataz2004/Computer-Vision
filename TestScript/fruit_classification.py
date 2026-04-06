import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "D:/Programs/PyCharm/pythonProject/Best Models/Fruit_Classification.pth"
INPUT_PATH = "D:/FCIS/Computer Vision/Project Data/Fruit/f"
OUTPUT_FILE = "PartC_Output.txt"
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint['classes']
num_classes = len(classes)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

if os.path.isdir(INPUT_PATH):
    image_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH)
                   if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))]
else:
    image_files = [INPUT_PATH]

results = []

with torch.no_grad():
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        output = model(img_tensor)
        pred = output.argmax(1).item()
        label = classes[pred]
        results.append(f"{os.path.basename(img_path)} -> {label}")

with open(OUTPUT_FILE, "w") as f:
    for line in results:
        f.write(line + "\n")

print(f"Finished. Results saved to {OUTPUT_FILE}")
