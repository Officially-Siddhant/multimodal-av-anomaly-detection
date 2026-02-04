# Multimodal Embodied AI for Anomaly Detection in Autonomous Vehicles

This project develops a multimodal anomaly detection framework for autonomous vehicles, integrating the [NVIDIA Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) Vision-Language Model for external semantic reasoning and an unsupervised Isolation Forest algorithm for internal fault monitoring.

The system processes data from cameras, LiDAR, and IMU sensors to detect environmental hazards and mechanical failures, such as fuel leaks and wheel misalignments. Validated using CARLA simulations and real-world datasets, the model achieved 100% recall for external anomaly detection on the Hazard Perception Test dataset and a mean inference time of 500 ms. Future work aims to improve precision and recall through enhanced dataset labeling schemes.

---

## Digital Twin for Synthetic Data Generation

This is developed in CARLA to simulate the ego vehicle, sensor suite, and surrounding environment, enabling controlled generation of diverse, safety-critical scenarios that are rare or impractical to capture in the real world. This simulation environment supports scalable data collection, sensor synchronization validation, and systematic stress-testing of multimodal anomaly detection under varied traffic, weather, and failure conditions, forming the foundation for robust model development and evaluation.

### Rigging Sample CAD Model

CARLA requires vehicle skeletons to be applied and configured onto the 3D model of the vehicle before custom import. One may follow [this](https://carla.readthedocs.io/en/latest/tuto_A_add_vehicle/) tutorial to add new vehicles.

Working with bones and pose alignment can be tricky; Blender is recommended for this task.

### Process in Blender

**Rigging Process:**

1. Identify moving and non-moving parts of the mesh file. Non-moving parts include the chassis, outer frame, and interior components. All wheel assemblies are moving parts.

2. Join all subcomponents as independent bodies (each wheel with its tyre and rim as one body). This results in 5 bodies total.

3. Attach bones to approximate cursor points for each component. These can be refined later.

4. Enter Object Mode, select a wheel → `Object > Set Origin > Origin to Geometry`.  
   Then apply transforms: `Ctrl + A > Rotation and Scale`.  
   The wheel rotation will now be correctly centered. Repeat for all wheels.

5. Select the wheel → `Shift + Select` the vehicle body → `Ctrl + P > Set Parent to Object`.  
   The wheel will now move correctly with the vehicle.