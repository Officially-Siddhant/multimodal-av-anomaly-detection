import cv2
import numpy as np
import os

def apply_faults(frame, fault_type="fog"):
    """
    Applies various synthetic visual faults to a frame.
    """
    if fault_type == "fog":
        # Simulate fog by increasing brightness and decreasing contrast
        fog_layer = np.full(frame.shape, 200, dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.4, fog_layer, 0.6, 0)
        # Add a blur to simulate visibility loss
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        
    elif fault_type == "blur":
        # Simulate a smudged or out-of-focus lens
        frame = cv2.medianBlur(frame, 15)
        
    elif fault_type == "noise":
        # Simulate sensor failure (Salt & Pepper noise)
        prob = 0.05
        thres = 1 - prob
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    frame[i][j] = [0, 0, 0]
                elif rdn > thres:
                    frame[i][j] = [255, 255, 255]
                    
    return frame

def generate_test_video(input_video, output_path):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating faulty video at: {output_path}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Start applying fault after the first 30 frames
        if frame_count > 30:
            frame = apply_faults(frame, fault_type="fog")
            
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print("✅ Done! Use this video in your run_inference() function.")

# RUN IT:
input_mp4 = r"C:\Users\swara\Desktop\AutonomousVehicle Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\dataset\Video_Dataset\test.mp4"
output_mp4 = "test_faulty_video.mp4"
generate_test_video(input_mp4, output_mp4)