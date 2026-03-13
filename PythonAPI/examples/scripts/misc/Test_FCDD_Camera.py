import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image

# =========================
# CONFIGURATION
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\fault_detector.pth"
TEST_VIDEO = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\dataset\Video_Dataset\1f47bf7f-d233-480c-b166-7512d8e9ac97.camera_front_wide_120fov.mp4" # Path to a video with fog/rain/damage

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# LOAD MODEL & CENTER
# =========================
class FCDD_Hardware(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 1, 1, bias=False) 
        )
    def forward(self, x): return self.features(x)

checkpoint = torch.load(MODEL_PATH)
model = FCDD_Hardware().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
c = checkpoint["center_c"].to(device)
model.eval()

# =========================
# LIVE TESTING LOOP
# =========================
cap = cv2.VideoCapture(TEST_VIDEO)

print("Starting Inference... Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Pre-process frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Calculate the Anomaly Map (distance from center c)
        dist_map = (output - c.view(1, 1, 1, 1)) ** 2
        anomaly_score = dist_map.mean().item()
        
        # 2. Process Heatmap for display
        heatmap = dist_map.squeeze().cpu().numpy()
        # Normalize heatmap for visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 3. Combine Original Frame + Heatmap
        overlay = cv2.addWeighted(frame, 0.7, heatmap_img, 0.3, 0)
        
        # 4. Display Logic
        # Adjust this threshold based on your 'NU' value
        threshold = 0.08 
        label = "FAULT DETECTED" if anomaly_score > threshold else "SYSTEM CLEAR"
        color = (0, 0, 255) if anomaly_score > threshold else (0, 255, 0)
        
        cv2.putText(overlay, f"{label} Score: {anomaly_score:.4f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Anomaly Detection Monitor", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# TEST FOR JPG

# import torch
# import torch.nn as nn
# from PIL import Image
# import torchvision.transforms as T

# # Setup device and path
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\fault_detector.pth"
# IMAGE_PATH = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\dataset\Kaggle Normal Data\images\1478019970188563338.jpg"

# # 1. Define the same architecture
# class FCDD_Hardware(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1, bias=False),
#             nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1, bias=False),
#             nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1, bias=False),
#             nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
#             nn.Conv2d(128, 1, 1, bias=False) 
#         )
#     def forward(self, x): return self.features(x)

# # 2. Load Model & Center
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# model = FCDD_Hardware().to(device)
# model.load_state_dict(checkpoint["model_state_dict"])
# c = checkpoint["center_c"].to(device)
# model.eval()

# # 3. Preprocess Image
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# img = Image.open(IMAGE_PATH).convert("RGB")
# img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension

# # 4. Get Score
# with torch.no_grad():
#     output = model(img_tensor)
#     dist_map = (output - c.view(1, 1, 1, 1)) ** 2
#     anomaly_score = dist_map.mean().item()

# print(f"Anomaly Score for {IMAGE_PATH}: {anomaly_score:.6f}")