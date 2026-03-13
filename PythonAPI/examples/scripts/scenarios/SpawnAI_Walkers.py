"""
CARLA Pedestrian Swarm & Random Crossing Simulation

OVERVIEW:
This script populates the CARLA environment with dynamic pedestrian traffic. 
It utilizes the 'Navigation' system to ensure walkers navigate the map 
intelligently while introducing randomized "road crossing" behaviors to 
simulate complex urban interactions.

KEY FUNCTIONAL BLOCKS:
1. SYNCHRONOUS WORLD CONTROL:
   - Configures CARLA to 'Synchronous Mode' with a fixed time-step (0.05s).
   - Ensures precise coordination between the Python client and the UE5 
     simulation engine, crucial for tracking high-speed physics or sensors.

2. BATCHED ACTOR SPAWNING:
   - Efficiently spawns a group of pedestrians ('Walkers') and their 
     associated 'AI Controllers' using CARLA’s batching API.
   - Pairs each walker with a controller actor to handle pathfinding and 
     speed modulation (randomized between 1.2 and 2.0 m/s).

3. DYNAMIC NAVIGATION & CROSSING:
   - Every simulation tick, there is a 2% probability for each pedestrian 
     to abandon their current path and initiate a new "crossing" goal.
   - Uses `get_random_location_from_navigation()` to ensure walkers move 
     to valid, reachable points across the map.

4. AUTOMATED CLEANUP:
   - Uses a 'finally' block to guarantee that all walkers and controllers 
     are destroyed upon exit (Ctrl+C), preventing "ghost actors" from 
     cluttering subsequent simulation runs.
"""
import sys
import glob
import os

# Modify this path to point to your CARLA folder
carla_path = glob.glob(
    r"C:\Users\swara\Desktop\CARLA_UE5\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
)
if carla_path:
    sys.path.append(carla_path[0])
else:
    raise FileNotFoundError("Couldn't find CARLA .egg file in your path")


import carla
import random
import time
import sys
import glob
import os

# --- Add CARLA egg to path ---
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# --- Connect to CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# --- Setup synchronous mode for better control ---
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# --- Pedestrian setup ---
num_walkers = 15
blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
controller_bp = world.get_blueprint_library().find('controller.ai.walker')

spawn_points = []
for i in range(num_walkers):
    loc = world.get_random_location_from_navigation()
    if loc:
        spawn_points.append(carla.Transform(loc))

# --- Spawn walkers ---
walker_batch = []
for spawn_point in spawn_points:
    bp = random.choice(blueprintsWalkers)
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'false')
    walker_batch.append(carla.command.SpawnActor(bp, spawn_point))

results = client.apply_batch_sync(walker_batch, True)
walker_ids = [r.actor_id for r in results if not r.error]

# --- Spawn walker controllers ---
controller_batch = []
for wid in walker_ids:
    controller_batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), wid))
controller_results = client.apply_batch_sync(controller_batch, True)
controller_ids = [r.actor_id for r in controller_results if not r.error]

# --- Collect all actors ---
all_ids = []
for i in range(len(walker_ids)):
    all_ids.append(controller_ids[i])
    all_ids.append(walker_ids[i])
all_actors = world.get_actors(all_ids)

# --- Start controllers ---
for i in range(0, len(all_ids), 2):
    controller = all_actors[i]
    walker = all_actors[i+1]
    controller.start()
    controller.set_max_speed(random.uniform(1.2, 2.0))  # walking speed

print(f"Spawned {num_walkers} walkers crossing roads randomly.")

# --- Random crossing behavior ---
try:
    while True:
        world.tick()

        for i in range(0, len(all_ids), 2):
            controller = all_actors[i]
            walker = all_actors[i+1]

            if random.random() < 0.02:  # 2% chance every tick to start new crossing
                start_loc = walker.get_location()
                end_loc = world.get_random_location_from_navigation()

                if end_loc is not None and start_loc.distance(end_loc) > 5.0:
                    # Make sure the walker crosses a road-like distance
                    controller.go_to_location(end_loc)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopping walkers...")

finally:
    # Stop controllers
    for i in range(0, len(all_ids), 2):
        all_actors[i].stop()

    # Destroy everything
    print("Cleaning up actors...")
    client.apply_batch([carla.command.DestroyActor(x) for x in all_ids])

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("Done.")
