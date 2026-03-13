"""

LiDAR Point Cloud 3D Viewer
This script provides a streamlined utility for visual inspection of LiDAR frames by converting raw NumPy data into an interactive 3D environment using Open3D.

"""
import numpy as np
import open3d as o3d

# Load your .npy file (LiDAR frame)
frame = np.load("dataset_lidar/anomaly/026923.npy")

# If the data includes intensity values, keep only XYZ
if frame.shape[1] > 3:
    frame = frame[:, :3]

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(frame)

# Open the 3D viewer
o3d.visualization.draw_geometries([pcd],
                                  window_name="LiDAR Frame Viewer",
                                  width=960,
                                  height=720,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
