"""
CARLA Wheel Alignment Monitoring & IMU Data Collection Suite

OVERVIEW:
This script simulates a vehicle dynamics monitoring system within the CARLA 
Simulator. It focuses on detecting wheel misalignment by comparing inertial 
data between different mounting points (wheel hubs vs. suspension arms).

WHAT THIS CODE DOES:
1.  ENVIRONMENT SETUP: Connects to a local CARLA server and spawns a 
    Tesla Model 3 with Autopilot enabled for consistent data collection.
    
2.  MISALIGNMENT INJECTION: 
    - Physically modifies the vehicle's physics control by injecting a 5-degree 
      toe-out error on the front-left wheel.
    - Artificially injects a 'Yaw Bias' into the front-left hub sensor data 
      stream to simulate sensor-level misalignment.

3.  MULTI-SENSOR ARRAY: Spawns and attaches eight (8) IMU sensors. 
    Each wheel is equipped with a pair of sensors:
    - HUB IMU: Mounted low, directly on the wheel hub.
    - ARM IMU: Mounted higher, on the suspension arm/chassis frame.

4.  DATA ACQUISITION: Streams real-time Accelerometer ($m/s^2$) and 
    Gyroscope ($rad/s$) data from all 8 sensors into a synchronized 
    dictionary, then exports the results to 'imu_data.csv'.

5.  ALIGNMENT ANALYSIS: 
    - Compares the Gyroscope data between 'Hub' and 'Arm' sensor pairs.
    - Calculates an 'Alignment Score' using the L2-norm of the mean 
      differential between sensor pairs.
    - Identifies and flags the specific wheel showing the highest 
      statistical deviation (the 'Misaligned' wheel).

REQUIREMENTS:
- CARLA Simulator (0.9.x)
- Dependencies: numpy, carla
"""

import random
import time
import os
import glob
import sys
from datetime import datetime
import numpy as np
import csv

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

# # Get a vehicle blueprint
# bp_lib = world.get_blueprint_library()
# vehicle_bp = bp_lib.find('vehicle.tesla.model3')

# # Spawn vehicle
# spawn_point = world.get_map().get_spawn_points()[0]
# vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# # --- Create IMU blueprints ---
# imu_bp = bp_lib.find('sensor.other.imu')

# # Function to spawn IMU at offset
# def attach_imu(vehicle, name, loc):
#     imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(**loc)), attach_to=vehicle)
#     imu.listen(lambda data: print(f"{name} Acc: {data.accelerometer}, Gyro: {data.gyroscope}"))
#     return imu

# # --- Attach 2 IMUs on front-left wheel ---
# imu1 = attach_imu(vehicle, "FL_hub_IMU", {"x": 1.0, "y": -0.8, "z": 0.3})
# imu2 = attach_imu(vehicle, "FL_arm_IMU", {"x": 1.0, "y": -0.8, "z": 0.6})

# # Let simulation run for a few seconds
# time.sleep(10)

# imu1.stop()
# imu2.stop()
# vehicle.destroy()


# ====================================================
# CONFIG
# ====================================================
SAVE_PATH = "imu_data.csv"
RUN_TIME = 20  # seconds
SHOW_OUTPUT = True

# # Misalignment simulation: yaw bias (radians)
# MISALIGNMENT_YAW = np.deg2rad(3.0)  # 3 degrees on front-left wheel

# # ====================================================
# # INITIALIZE CARLA
# # ====================================================
# client = carla.Client("localhost", 2000)
# client.set_timeout(10.0)
# world = client.get_world()

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.tesla.model3')

spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

physics_control = vehicle.get_physics_control()
wheels = physics_control.wheels

# Introduce camber/toe misalignment on front-left wheel
wheels[2].steer_angle = 5.0  # degrees toe-out
physics_control.wheels = wheels

vehicle.apply_physics_control(physics_control)

print("✅ Applied 5° front-left wheel misalignment")

# ====================================================
# SETUP IMUs
# ====================================================
imu_bp = bp_lib.find('sensor.other.imu')

imu_offsets = {
    "FL_hub": carla.Location(x=1.0, y=-0.8, z=0.3),
    "FL_arm": carla.Location(x=1.0, y=-0.8, z=0.6),
    "FR_hub": carla.Location(x=1.0, y=0.8, z=0.3),
    "FR_arm": carla.Location(x=1.0, y=0.8, z=0.6),
    "RL_hub": carla.Location(x=-1.0, y=-0.8, z=0.3),
    "RL_arm": carla.Location(x=-1.0, y=-0.8, z=0.6),
    "RR_hub": carla.Location(x=-1.0, y=0.8, z=0.3),
    "RR_arm": carla.Location(x=-1.0, y=0.8, z=0.6),
}

imus = {}
imu_data = {name: [] for name in imu_offsets.keys()}


def imu_callback(name):
    def callback(data):
        t = data.timestamp
        accel = [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z]
        gyro = [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z]

        # Apply artificial misalignment on FL hub
        if "FL_hub" in name:
            gyro = [gyro[0], gyro[1], gyro[2] + MISALIGNMENT_YAW]

        imu_data[name].append([t] + accel + gyro)
        if SHOW_OUTPUT:
            print(f"{name}: gyro={gyro}")
    return callback


for name, loc in imu_offsets.items():
    imu = world.spawn_actor(imu_bp, carla.Transform(loc), attach_to=vehicle)
    imu.listen(imu_callback(name))
    imus[name] = imu


print("Running simulation... collecting IMU data.")
time.sleep(RUN_TIME)

# ====================================================
# SAVE IMU DATA
# ====================================================
print("\nSaving IMU data to CSV...")
with open(SAVE_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["sensor", "timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
    for name, readings in imu_data.items():
        for row in readings:
            writer.writerow([name] + row)

# ====================================================
# CLEANUP
# ====================================================
for imu in imus.values():
    imu.stop()
    imu.destroy()

vehicle.destroy()
print("Data saved to:", SAVE_PATH)

# ====================================================
# SIMPLE ALIGNMENT ANALYSIS
# ====================================================
def compute_alignment_score(sensor1, sensor2):
    data1 = np.array(imu_data[sensor1])
    data2 = np.array(imu_data[sensor2])
    if len(data1) == 0 or len(data2) == 0:
        return None
    min_len = min(len(data1), len(data2))
    gyro_diff = np.mean(np.abs(data1[:min_len, 4:7] - data2[:min_len, 4:7]), axis=0)
    return np.linalg.norm(gyro_diff)

pairs = [("FL_hub", "FL_arm"), ("FR_hub", "FR_arm"), ("RL_hub", "RL_arm"), ("RR_hub", "RR_arm")]

scores = {}
for s1, s2 in pairs:
    score = compute_alignment_score(s1, s2) or 0.0
    scores[f"{s1}-{s2}"] = score

print("\n--- Wheel Alignment Scores ---")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

misaligned = max(scores, key=scores.get)
print(f"\n⚠️ Detected likely misalignment at: {misaligned}")
