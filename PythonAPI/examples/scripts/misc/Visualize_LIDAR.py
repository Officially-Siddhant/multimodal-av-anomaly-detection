"""
LiDAR Anomaly Detection CLI: Training & Inference Pipeline

OVERVIEW:
This script provides a command-line interface (CLI) to train and deploy a Deep 
Learning Autoencoder for detecting anomalies in LiDAR point clouds. It identifies 
structural deviations in 3D sensor data by calculating reconstruction fidelity.

CORE CAPABILITIES:
1. ARGUMENT PARSING:
   - Supports --mode 'train' or 'test' for switching between development and deployment.
   - Allows custom --threshold settings for sensitivity tuning and specific --file input.

2. TRAINING MODE:
   - Processes raw .npy LiDAR frames from 'dataset_lidar/normal/'.
   - Normalizes input by padding/clipping to a fixed 3,200 points (12,800 features).
   - Trains a deep feed-forward Autoencoder with BatchNorm and ReLU activations 
     to establish a "Normalcy" baseline.



3. TESTING & ANOMALY DETECTION:
   - Loads the pre-trained 'lidar_autoencoder.pth' model.
   - Calculates Mean Squared Error (MSE) between the input frame and its 
     reconstructed counterpart.
   - Classification: MSE > threshold = 'Anomaly'; MSE <= threshold = 'Normal'.

4. 3D HEATMAP VISUALIZATION:
   - Generates a Matplotlib 3D scatter plot of the point cloud.
   - Colors points based on their individual reconstruction error (Euclidean distance).
   - 'Coolwarm' colormap highlights specific areas of the point cloud that the 
     model failed to reconstruct, pinpointing the physical location of the anomaly.



REQUIREMENTS:
- CARLA Simulator environment for data generation.
- PyTorch, NumPy, Matplotlib.
"""
import os
import glob
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
# 1. Argument Parser
# ============================================================
parser = argparse.ArgumentParser(description="LiDAR Anomaly Detection")
parser.add_argument("--mode", choices=["train", "test"], required=True,
                    help="Choose 'train' to train model or 'test' to detect anomalies.")
parser.add_argument("--file", type=str, default=None,
                    help="Path to .npy file for anomaly testing (optional).")
parser.add_argument("--threshold", type=float, default=1.0,
                    help="Anomaly score threshold (default=1.0).")
args = parser.parse_args()

# ============================================================
# 2. Define Autoencoder
# ============================================================
class LiDARAutoencoder(nn.Module):
    def __init__(self, input_dim=12800):
        super(LiDARAutoencoder, self).__init__()
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
# 3. Device setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

# ============================================================
# 4. Training mode
# ============================================================
if args.mode == "train":
    print("\n📂 Loading training data from dataset_lidar/normal/ ...")
    train_data = []
    os.makedirs("dataset_lidar/normal", exist_ok=True)

    for f in os.listdir('dataset_lidar/normal'):
        if f.endswith('.npy'):
            pc = np.load(os.path.join('dataset_lidar/normal', f))
            if pc.shape[0] < 3200:
                pc = np.pad(pc, ((0, 3200 - pc.shape[0]), (0, 0)))
            pc = pc[:3200, :].flatten()
            train_data.append(pc)

    if len(train_data) == 0:
        raise FileNotFoundError("❌ No .npy files found in dataset_lidar/normal/. Record data first!")

    X = torch.tensor(train_data, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X, batch_size=16, shuffle=True)
    print(f"✅ Loaded {len(train_data)} LiDAR frames for training.")

    model = LiDARAutoencoder(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n🚀 Training Autoencoder...")
    for epoch in range(20):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1:02d}/20 | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "lidar_autoencoder.pth")
    print("💾 Model saved as lidar_autoencoder.pth")
    sys.exit(0)

# ============================================================
# 5. Testing mode
# ============================================================
if args.mode == "test":
    print("\n🧠 Loading trained model...")
    model = LiDARAutoencoder(input_dim=12800).to(device)
    if not os.path.exists("lidar_autoencoder.pth"):
        raise FileNotFoundError("❌ Model file 'lidar_autoencoder.pth' not found! Train the model first.")
    model.load_state_dict(torch.load("lidar_autoencoder.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    def lidar_anomaly_score(file_path, threshold=args.threshold):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        pc = np.load(file_path)
        if pc.shape[0] < 3200:
            pc = np.pad(pc, ((0, 3200 - pc.shape[0]), (0, 0)))
        pc = pc[:3200, :]

        x_flat = pc.flatten()
        x = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)
            loss = criterion(recon, x).item()
            recon_pc = recon.cpu().numpy().reshape(-1, 4)

        print(f"\n🔍 File: {os.path.basename(file_path)}")
        print(f"Anomaly Score: {loss:.5f}")
        if loss > threshold:
            print("⚠️ Detected Anomaly")
        else:
            print("✅ Normal Frame")

        # Visualization
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        diffs = np.linalg.norm(pc[:, :3] - recon_pc[:, :3], axis=1)
        normalized_diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min() + 1e-8)

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   c=normalized_diffs, cmap='coolwarm', s=2)

        ax.set_title(f"LiDAR Anomaly Visualization\nScore={loss:.5f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    # Pick file automatically if not provided
    if args.file:
        test_file = args.file
    else:
        normal_files = glob.glob("dataset_lidar/normal/*.npy")
        if not normal_files:
            raise FileNotFoundError("❌ No normal LiDAR data found for testing.")
        test_file = random.choice(normal_files)
        print(f"📁 Automatically selected test file: {test_file}")

    lidar_anomaly_score(test_file, threshold=args.threshold)
    print("\n✅ Test completed successfully.")
