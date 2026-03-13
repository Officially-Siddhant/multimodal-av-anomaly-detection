"""
FCDD Visual Anomaly Inference & Heatmap Localization

OVERVIEW:
This script performs inference using a trained Fully Convolutional Data Description 
(FCDD) model to detect and spatially localize anomalies in vehicle camera images. 
Unlike global classifiers, this script identifies exactly which regions of an 
image deviate from the "normal" training distribution.

HOW IT WORKS:
1. SPATIAL FEATURE MAPPING:
   - The model processes a 224x224 input image through a fully convolutional 
     network, resulting in a downsampled feature map where each pixel represents 
     a local receptive field of the original scene.

2. EUCLIDEAN DEVIATION SCORING:
   - For every spatial location in the output map, the script calculates the 
     squared Euclidean distance from the pre-learned hypersphere center 'c':
     $$AnomalyScore = (y_{i,j} - c)^2$$
   - Points close to 'c' represent normal road features, while distant points 
     indicate anomalies (e.g., obstacles, glare, or sensor artifacts).



3. HEATMAP OVERLAY:
   - The raw anomaly scores are converted into a 'Jet' colormap and 
     superimposed onto the original image using alpha-blending.
   - High-intensity regions (red) pinpoint the exact coordinates of 
     detected anomalies, while low-intensity regions (blue) indicate 
     normality.



USAGE:
- Requires 'fcdd_camera.pth' (containing both 'model_state_dict' and 'center_c').
- Input: Individual PNG/JPG frames from the CARLA simulator or real-world datasets.
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Transform (must match training)
# =========================
camera_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# =========================
# Load image
# =========================
image_path = "dataset/FCDD_test_data/018966.png"  # change to your image
img = Image.open(image_path).convert("RGB")
x = camera_transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# =========================
# Load trained FCDD model
# =========================
class FCDD_Camera(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 1, 1, bias=False)
        )

    def forward(self, x):
        return self.net(x)

checkpoint = torch.load("fcdd_camera.pth", map_location=device)
model = FCDD_Camera().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
c = checkpoint["center_c"].to(device)

# =========================
# Forward pass and heatmap
# =========================
with torch.no_grad():
    y = model(x)  # [1, 1, H, W]

    # Compute squared deviation from center
    dist_map = (y - c.view(1,1,1,1))**2  # [1,1,H,W]
    heatmap = dist_map[0,0].cpu().numpy()

# =========================
# Visualize heatmap
# =========================
plt.figure(figsize=(6,6))
plt.imshow(np.array(img.resize((heatmap.shape[1], heatmap.shape[0]))))  # original image
plt.imshow(heatmap, cmap='jet', alpha=0.5)  # overlay heatmap
plt.colorbar(label="Anomaly Score")
plt.title("FCDD Anomaly Heatmap")
plt.axis('off')
plt.show()
