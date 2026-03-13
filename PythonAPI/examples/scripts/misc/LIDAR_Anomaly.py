"""
CARLA LiDAR Anomaly Detection via Deep Autoencoder

OVERVIEW:
This script performs real-time anomaly detection on vehicle LiDAR point clouds 
using a PyTorch-based Deep Autoencoder. It identifies "anomalous" frames 
(e.g., sensor noise, unexpected obstacles, or hardware failure) based on 
reconstruction error.

CORE LOGIC:
1. ARCHITECTURE: Implements a symmetric Encoder-Decoder network. The Encoder 
   compresses 12,800-dimensional flattened point clouds into a 128-unit 
   latent representation, while the Decoder attempts to reconstruct the input.
   
2. DATA PROCESSING: 
   - Normalizes input point clouds to a fixed size (3,200 points).
   - Flattens spatial coordinates (x, y, z, intensity) for neural processing.

3. ANOMALY HEURISTIC: 
   - Uses Mean Squared Error (MSE) to measure the difference between the 
     original frame and its reconstruction.
   - High MSE (Reconstruction Loss) indicates the model does not recognize 
     the pattern, flagging it as an 'Anomaly'.

USAGE:
Requires 'lidar_autoencoder.pth'. Evaluates a random sample from the 
'dataset_lidar/anomaly/' directory and prints the anomaly score.
"""

import random
import time
import os
import glob
import sys
import torch
import torch.nn as nn
import numpy as np
import random

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
# ============================================================
# 1. Define Autoencoder
# ============================================================
class LiDARAutoencoder(nn.Module):
    def __init__(self, input_dim=12800):
        # super(LiDARAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ============================================================
# 2. Load model weights
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LiDARAutoencoder(input_dim=12800).to(device)

if not os.path.exists("lidar_autoencoder.pth"):
    raise FileNotFoundError("❌ Model file 'lidar_autoencoder.pth' not found! Train the model first.")

model.load_state_dict(torch.load("lidar_autoencoder.pth", map_location=device))
model.eval()
print("✅ Loaded trained model from lidar_autoencoder.pth")

criterion = nn.MSELoss()

# ============================================================
# 3. Define Anomaly Detection Function
# ============================================================
def lidar_anomaly_score(file_path, threshold=0.02):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    pc = np.load(file_path)
    if pc.shape[0] < 3200:
        pc = np.pad(pc, ((0, 3200 - pc.shape[0]), (0, 0)))
    pc = pc[:3200, :].flatten()

    x = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
        loss = criterion(recon, x).item()
        

    print(f"\n🔍 File: {os.path.basename(file_path)}")
    print(f"Anomaly Score: {loss:.5f}")
    if loss > threshold:
         print("⚠️ Detected Anomaly")
    else:
        print("✅ Normal Frame")

# ============================================================
# 4. Automatically test a sample file
# ============================================================
normal_files = glob.glob("dataset_lidar/anomaly/*.npy")
if not normal_files:
    raise FileNotFoundError("❌ No normal LiDAR data found in dataset_lidar/anomaly/. Please record or copy .npy files there.")

test_file = random.choice(normal_files)
print(f"\n📁 Testing on: {test_file}")
lidar_anomaly_score(test_file, threshold=1.0)

print("\n✅ Script finished successfully.")
