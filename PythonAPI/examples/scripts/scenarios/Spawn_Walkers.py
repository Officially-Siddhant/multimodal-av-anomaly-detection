"""
CARLA Manual Pedestrian Kinematics & Random Crossing Simulation

OVERVIEW:
This script bypasses the standard CARLA AI Controller to move pedestrians 
using custom kinematic equations. It creates a "swarm" of walkers that 
manually navigate towards randomized targets, providing more granular 
control over individual movement patterns and crossing timing.

KEY COMPONENTS:
1. MANUAL KINEMATICS ENGINE:
   - Instead of using `controller.ai.walker`, the script uses a custom 
     `move_towards` function. 
   - It calculates a 2D direction vector ($direction = target - current$) 
     and updates the actor's position using: 
     $loc = loc + (direction \times speed \times dt)$.

2. STATE MACHINE ARCHITECTURE:
   - Each pedestrian is assigned a state dictionary tracking:
     * Movement status (moving/idle)
     * Target destination (sampled from the Navigation Mesh)
     * Randomized walking speed (1.0 to 2.0 m/s)
     * Cooldown timers to prevent constant, robotic movement.

3. SPATIAL LOGIC:
   - Uses `get_random_location_from_navigation()` to ensure target points 
     are valid pedestrian areas, but performs manual distance checks 
     (> 8.0 meters) to ensure "crossing" events are substantial.

4. SYNCHRONOUS ORCHESTRATION:
   - Locks the simulation to 20 FPS (fixed delta of 0.05s) to ensure 
     the manual position updates remain smooth and deterministic across 
     the entire swarm.
"""

import math
import random
import time
import sys
import glob
import os

# --- Add CARLA egg to path ---
carla_egg = glob.glob(
    r"C:\Users\swara\Desktop\CARLA_UE5\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
)
if carla_egg:
    sys.path.append(carla_egg[0])
else:
    raise FileNotFoundError("Couldn't find CARLA .egg file in your path")

import carla

# --- Connect to CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# --- Setup synchronous mode ---
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# --- Spawn pedestrians (NO AI) ---
num_walkers = 100
blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

spawn_points = []
for i in range(num_walkers):
    loc = world.get_random_location_from_navigation()
    if loc:
        spawn_points.append(carla.Transform(loc))

walker_batch = []
for spawn_point in spawn_points:
    bp = random.choice(blueprintsWalkers)
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'false')
    walker_batch.append(carla.command.SpawnActor(bp, spawn_point))

results = client.apply_batch_sync(walker_batch, True)
walker_ids = [r.actor_id for r in results if not r.error]
walkers = world.get_actors(walker_ids)

print(f"Spawned {len(walkers)} pedestrians (no AI).")

# --- Each walker has a crossing timer and state ---
walker_state = {
    walker.id: {
        "moving": False,
        "target": None,
        "speed": random.uniform(1.0, 2.0),  # m/s
        "next_cross_time": time.time() + random.uniform(2, 8)
    }
    for walker in walkers
}

# --- Utility: move pedestrian manually ---
def move_towards(walker, target, speed, dt):
    loc = walker.get_location()
    direction = carla.Vector3D(
        target.x - loc.x,
        target.y - loc.y,
        target.z - loc.z
    )
    dist = math.sqrt(direction.x**2 + direction.y**2)
    if dist < 0.1:
        return True  # reached

    direction.x /= dist
    direction.y /= dist
    loc.x += direction.x * speed * dt
    loc.y += direction.y * speed * dt
    walker.set_location(loc)
    return False

# --- Main loop ---
try:
    print("Pedestrians crossing roads randomly...")
    while True:
        world.tick()
        now = time.time()

        for walker in walkers:
            state = walker_state[walker.id]

            # Start a new crossing occasionally
            if not state["moving"] and now > state["next_cross_time"]:
                start = walker.get_location()
                end = world.get_random_location_from_navigation()

                if end and start.distance(end) > 8.0:
                    state["moving"] = True
                    state["target"] = end
                    state["speed"] = random.uniform(1.0, 2.0)
                else:
                    # Retry later
                    state["next_cross_time"] = now + random.uniform(2, 8)

            # Move active walkers
            if state["moving"]:
                reached = move_towards(
                    walker, state["target"], state["speed"], settings.fixed_delta_seconds
                )
                if reached:
                    state["moving"] = False
                    state["target"] = None
                    # Wait random seconds before next crossing
                    state["next_cross_time"] = now + random.uniform(3, 10)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping pedestrians...")

finally:
    print("Cleaning up actors...")
    client.apply_batch([carla.command.DestroyActor(x) for x in walker_ids])

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("Done.")