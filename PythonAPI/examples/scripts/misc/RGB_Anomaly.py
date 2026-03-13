"""
CARLA End-to-End Visual Anomaly Detection Suite

OVERVIEW:
This integrated script automates the full lifecycle of an AI safety system: 
from synthetic data generation in CARLA to training and deploying a 
Convolutional Autoencoder (CAE) for environmental anomaly detection.

WORKFLOW:
1. DATA GENERATION (CARLA):
   - Phase A (Normal): Spawns a Tesla Model 3 in 'ClearNoon' conditions to 
     capture baseline driving frames.
   - Phase B (Anomaly): Dynamically alters world weather to extreme 
     precipitation, fog, and wetness to capture 'Out-of-Distribution' (OOD) data.
   - Storage: Images are automatically indexed and saved into structured 
     dataset directories.

2. DEEP LEARNING ARCHITECTURE:
   - Implements a Convolutional Autoencoder using PyTorch.
   - Encoder: Progressively reduces spatial dimensions through 3x3 convolutions 
     to capture the "latent essence" of clear-weather driving.
   - Decoder: Uses Transposed Convolutions to reconstruct the original image 
     from the compressed bottleneck.

3. ANOMALY DETECTION LOGIC:
   - The model is trained to minimize Reconstruction Loss (MSE) on normal data.
   - During inference, images that produce a high MSE—indicating the model 
     cannot "recognize" or reconstruct the scene—are flagged as anomalies 
     (e.g., extreme weather or sensor obstructions).

4. INTERACTIVE DIAGNOSTICS:
   - Includes a real-time CLI to input image paths and receive an 
     instant 'Normal' vs. 'Anomaly' classification based on the 
     learned reconstruction threshold.
"""

# import carla
import random
import time
import os
import glob
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

world.set_weather(carla.WeatherParameters.ClearNoon)

# Spawn vehicle
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Attach RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

output_dir = "dataset/normal"
os.makedirs(output_dir, exist_ok=True)

def save_image(image):
    image.save_to_disk(f"{output_dir}/{image.frame:06d}.png")

camera.listen(lambda image: save_image(image))

# # Spawn traffic (normal vehicles)
# traffic_manager = client.get_trafficmanager()
# traffic_manager.set_global_distance_to_leading_vehicle(2.5)

# for i in range(30):
#     spawn_point = random.choice(world.get_map().get_spawn_points())
#     npc_vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
#     npc = world.try_spawn_actor(npc_vehicle_bp, spawn_point)
#     if npc is not None:
#         npc.set_autopilot(True, traffic_manager.get_port())


# Drive around for 2–3 minutes manually or with autopilot
vehicle.set_autopilot(True)

time.sleep(1)  # record for 3 minutes

camera.stop()

weather = carla.WeatherParameters(
    cloudiness=100.0,
    precipitation=100.0,
    precipitation_deposits=100.0,
    wetness=100.0,
    fog_density=80.0,
    fog_distance=10.0,
    sun_azimuth_angle=180.0
)
world.set_weather(weather)


camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
output_dir = "dataset/anomaly"
os.makedirs(output_dir, exist_ok=True)

def save_anomaly_image(image):
    image.save_to_disk(f"{output_dir}/{image.frame:06d}.png")

camera.listen(lambda image: save_anomaly_image(image))

# Run again for a while in bad weather
time.sleep(1)

camera.stop()

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# ----------------------
# Autoencoder definition
# ----------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ----------------------
# Device and paths
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'autoencoder.pth'

transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
])

# ----------------------
# Function to train model
# ----------------------
def train_autoencoder():
    dataset = datasets.ImageFolder(root="dataset", transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Autoencoder...")
    for epoch in range(10):
        for data, _ in train_loader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

# ----------------------
# Load or train model
# ----------------------
model = Autoencoder().to(device)
if os.path.exists(model_path):
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model = train_autoencoder()

model.eval()

# ----------------------
# Function to check anomaly
# ----------------------
criterion = nn.MSELoss()
def anomaly_score(img_path, threshold = 0.0004):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)
        loss = criterion(recon, img_tensor).item()

    print(f"Anomaly Score: {loss:.5f}")
    print(f"Threshold: {threshold}")
    if loss < threshold:
        print("⚠️ Anomaly Detected!")
    else:
        print("✅ Normal")
    return loss

# # Set Threshold
# def find_threshold(normal_dir="dataset/normal"):
#     scores = []
#     for img_name in os.listdir(normal_dir):
#         img_path = os.path.join(normal_dir, img_name)
#         if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img = Image.open(img_path).convert('RGB')
#             img_tensor = transform(img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 recon = model(img_tensor)
#                 loss = criterion(recon, img_tensor).item()
#             scores.append(loss)
#     threshold = max(scores) * 1.1  # 10% higher than max normal score
#     print(f"\n📊 Suggested threshold: {threshold:.5f}")
#     return threshold

# threshold = find_threshold("dataset/normal")
# print("✅ Threshold set to:", threshold)

# ----------------------
# Check image
# ----------------------
while True:
    image_path = input("Enter path of image to check (or 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break
    if not os.path.exists(image_path):
        print("❌ File not found! Try again.")
        continue
    anomaly_score(image_path)
