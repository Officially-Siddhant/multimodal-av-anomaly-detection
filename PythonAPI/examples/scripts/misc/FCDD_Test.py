import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# =========================
# 1. Device & Configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
IMG_SIZE = 224
VIDEO_DIR = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\dataset\Video_Dataset"  # Path to your .mp4 files
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-6
NU = 0.001  # Sensitivity margin
FRAME_SKIP = 10  # Sample every 10th frame

camera_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 2. Video Dataset
# =========================
class VideoFaultDataset(Dataset):
    def __init__(self, video_dir, transform, frame_interval=10):
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                           if f.lower().endswith((".mp4", ".avi", ".mov"))]
        self.transform = transform
        self.frame_interval = frame_interval
        self.samples = []
        
        print(f"Indexing videos in {video_dir}...")
        for v_path in self.video_files:
            cap = cv2.VideoCapture(v_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, frame_count, self.frame_interval):
                self.samples.append((v_path, i))
            cap.release()
        print(f"Total frames sampled for training: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_path, frame_idx = self.samples[idx]
        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return self.transform(img)

# =========================
# 3. FCDD Model Architecture
# =========================
class FCDD_Hardware(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: Captures basic edges/textures
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Layer 2: Captures complex patterns
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Layer 3: High-level spatial features
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            
            # Final Mapping: 1x1 conv to produce the anomaly map
            nn.Conv2d(128, 1, 1, bias=False) 
        )

    def forward(self, x):
        return self.features(x)

# =========================
# 4. Training Loop
# =========================
def train():
    dataset = VideoFaultDataset(VIDEO_DIR, camera_transform, FRAME_SKIP)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = FCDD_Hardware().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    # Initialize hypersphere center 'c'
    print("Initializing center c...")
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            if i > 5: break # Sample first few batches for center init
            y = model(x.to(device))
            all_outputs.append(y.mean(dim=[0, 2, 3]))
    c = torch.stack(all_outputs).mean(dim=0).detach()
    print(f"Center 'c' initialized at: {c.item():.4f}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x in dataloader:
            x = x.to(device)
            y = model(x)
            
            # FCDD Loss: Mean of (dist^2) with a hinge at NU
            dist = (y - c.view(1, 1, 1, 1)) ** 2
            score = dist.mean(dim=[2, 3]) 
            loss = torch.mean(torch.maximum(score, torch.tensor(NU).to(device)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {total_loss/len(dataloader):.6f}")

    # Save everything
    torch.save({
        "model_state_dict": model.state_dict(),
        "center_c": c
    }, "fault_detector.pth")
    print("✅ Model Saved.")

# =========================
# 5. Inference & Heatmap Overlay
# =========================
def run_inference(video_path, model_path="fault_detector.pth"):
    checkpoint = torch.load(model_path)
    model = FCDD_Hardware().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    c = checkpoint["center_c"].to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Prepare frame
        input_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = camera_transform(input_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Calculate distance map
            dist_map = (output - c.view(1, 1, 1, 1)) ** 2
            anomaly_score = dist_map.mean().item()
            
            # Resize distance map to original frame size for visualization
            heatmap = dist_map.squeeze().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay heatmap on original frame
            overlay = cv2.addWeighted(frame, 0.6, heatmap_img, 0.4, 0)
            
            # Alert Text
            color = (0, 0, 255) if anomaly_score > NU * 2 else (0, 255, 0)
            status = "FAULT DETECTED" if anomaly_score > NU * 2 else "CAMERA CLEAR"
            cv2.putText(overlay, f"{status} ({anomaly_score:.4f})", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("Lens Fault Detection", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment train() to run training
    train()

    # Run inference on a test video
    # run_inference("dataset/test_foggy_video.mp4")
    pass

# import torch
# print(f"Is CUDA available? {torch.cuda.is_available()}")
# print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")