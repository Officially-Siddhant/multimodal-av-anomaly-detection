"""
CARLA Autonomous Safety: Visual Anomaly Detection Pipeline

OVERVIEW:
This script implements a complete machine learning lifecycle for autonomous 
driving safety. It utilizes the CARLA simulator to generate data and a PyTorch 
Convolutional Autoencoder to detect "out-of-distribution" environmental 
anomalies (e.g., severe weather) that could impair vehicle perception.

WORKFLOW:
1. DATA ACQUISITION (CARLA):
   - Baseline: Captures "Normal" driving frames under 'ClearNoon' weather.
   - Stress Test: Captures "Anomaly" frames by simulating extreme conditions 
     (100% precipitation, 80% fog density, and high wetness).
   - Sensor: Utilizes a 90° FOV RGB camera mounted to an ego-vehicle 
     operating on Autopilot.

2. NEURAL NETWORK ARCHITECTURE:
   - Convolutional Autoencoder: Features a 3-layer Encoder to compress visual 
     input and a 3-layer Decoder (Transposed Convolutions) to reconstruct it.
   - Objective: Learns to perfectly reconstruct "Normal" weather patterns, 
     resulting in low Mean Squared Error (MSE).

3. STATISTICAL ANOMALY DETECTION:
   - Scoring: Calculates the reconstruction loss (MSE) for individual frames.
   - Thresholding: Establishes an 'Empirical Threshold' by calculating the 
     midpoint between the highest normal score and the lowest anomaly score.
   - Classification: Any frame yielding a score above this threshold is 
     flagged as a perceptual anomaly, alerting the system to unsafe conditions.



USAGE:
Requires a running CARLA server. The script executes data collection, model 
training, and threshold evaluation sequentially.
"""

import random
import time
import os
import glob
import sys

# --- Setup CARLA ---
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import sys

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# --- Spawn ego vehicle ---
vehicle_bp = bp_lib.filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

# --- Camera setup ---
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# --- Directories ---
normal_dir = "dataset/normal"
anomaly_dir = "dataset/anomaly"
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

# --- Save function ---
def save_image_normal(image):
    image.save_to_disk(f"{normal_dir}/{image.frame:06d}.png")

def save_image_anomaly(image):
    image.save_to_disk(f"{anomaly_dir}/{image.frame:06d}.png")

# --- Collect normal images ---
world.set_weather(carla.WeatherParameters.ClearNoon)
camera.listen(save_image_normal)
print("Collecting normal images for 3 minutes...")
time.sleep(180)
camera.stop()

# --- Collect anomaly images (heavy weather) ---import torch
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor()
])

# --- Dataset ---
dataset = datasets.ImageFolder(root="dataset", transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
epochs = 10
for epoch in range(epochs):
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon = model(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'autoencoder.pth')

# --- Anomaly scoring function ---
def anomaly_score(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        recon = model(img_tensor)
        loss = criterion(recon, img_tensor).item()
    return loss

# --- Compute empirical threshold ---
normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
anomaly_images = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)]

normal_scores = [anomaly_score(f) for f in normal_images[:100]]
anomaly_scores = [anomaly_score(f) for f in anomaly_images[:100]]

threshold = max(normal_scores) + (min(anomaly_scores) - max(normal_scores))/2
print("Empirical Threshold:", threshold)

# --- Test a single image ---
test_image = anomaly_images[0]
score = anomaly_score(test_image)
print("Image:", test_image, "Score:", score)
if score > threshold:
    print("⚠️ Anomaly Detected!")
else:
    print("Normal")

heavy_weather = carla.WeatherParameters(
    cloudiness=100.0,
    precipitation=100.0,
    precipitation_deposits=100.0,
    wetness=100.0,
    fog_density=80.0,
    fog_distance=10.0,
    sun_azimuth_angle=180.0
)
world.set_weather(heavy_weather)
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(save_image_anomaly)
print("Collecting anomaly images for 3 minutes...")
time.sleep(180)
camera.stop()

print("Data collection complete!")

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor()
])

# --- Dataset ---
dataset = datasets.ImageFolder(root="dataset", transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
epochs = 10
for epoch in range(epochs):
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon = model(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'autoencoder.pth')

# --- Anomaly scoring function ---
def anomaly_score(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        recon = model(img_tensor)
        loss = criterion(recon, img_tensor).item()
    return loss

# --- Compute empirical threshold ---
normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
anomaly_images = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)]

normal_scores = [anomaly_score(f) for f in normal_images[:100]]
anomaly_scores = [anomaly_score(f) for f in anomaly_images[:100]]

threshold = max(normal_scores) + (min(anomaly_scores) - max(normal_scores))/2
print("Empirical Threshold:", threshold)

# --- Test a single image ---
test_image = anomaly_images[0]
score = anomaly_score(test_image)
print("Image:", test_image, "Score:", score)
if score > threshold:
    print("⚠️ Anomaly Detected!")
else:
    print("Normal")


