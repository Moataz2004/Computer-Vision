import os
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "D:/Programs/PyCharm/pythonProject/Best Models/FoodVsFruit_StageA.pth"
input_path = r"D:/FCIS/Computer Vision/Project Data/Fruit/f"
output_file = "PartA_output.txt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

if os.path.isdir(input_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]
else:
    image_files = [input_path]

results = []

with torch.no_grad():
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        pred = output.argmax(1).item()
        label = "Food" if pred == 0 else "Fruit"
        results.append(f"{os.path.basename(img_path)} -> {label}")

with open(output_file, "w") as f:
    for line in results:
        f.write(line + "\n")
print(f"Finished. Results saved to {output_file}")
