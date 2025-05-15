# ROS2 PX4 Rescue Challenge

![Drone over lake image](/images/uav_lake.jpeg)

## Project Overview

This project implements an autonomous UAV mission for the Lake Rescue Challenge using ROS 2 and PX4. The solution uses computer vision to detect people in a simulated environment and reports their locations.

### Key Components
- **avader/**: Challenge framework providing scoring and simulation environment
- **solver/**: Our implementation for detection, navigation, and reporting

### Main Functions in `challenge.py`
- **Initialization**: Sets up publishers, subscribers, and mission parameters
- **Mission Control**: Handles mission start, trajectory planning, and navigation
- **Object Detection**: Processes camera feed to identify and locate people
- **Reporting**: Publishes detected people count and their precise locations

## About the Challenge

The Lake Rescue Challenge simulates a rescue mission where a UAV (Unmanned Aerial Vehicle) assists search and rescue teams in locating stranded individuals on a lake. Participants must pilot the drone to find people scattered across the mission area and relay their precise locations. Using ROS 2 Humble, participants are tasked with meeting strict criteria for accuracy, speed, and data reporting.

## Scoring System

The maximum achievable score is **100 points**. Points are awarded based on the following four criteria:

### 1. Number of Points Visited (40 Points)

Coordinates of target points in 3D space will be published on the `/avader/locations_to_visit` topic with `geometry_msgs/msg/PoseArray` as the message type. Participants must guide the UAV to each of these locations with a tolerance of 0.4 meters.

- **Scoring**: 40 points divided by the total number of points to be visited. For example, if there are 10 points, each visited point will earn 4 points.

### 2. Accurate Object Count (20 Points)

During the UAV's flight, objects will be visible on the camera feed available on the `/camera` topic. The participant's task is to count the total number of visible objects in the entire search area and publish the count to the `/avader/people_count` topic using `std_msgs/msg/Int32`.

- **Scoring**: 20 points for correctly reporting the total object count.

### 3. Reporting Object Locations (30 Points)

Participants must publish the location of each detected person in the local NED coordinate frame to the `/avader/location_objects` topic. The `local NED` origin is defined as the UAV’s starting position, and object locations should be provided with an accuracy of 0.8 meters. Each object’s position should assume a height of `Z = 0`.

- **Scoring**: 30 points divided by the total number of objects; for example, if there are 6 objects, each correctly reported location will award 5 points.

### 4. Flight Time (10 Points)

Faster completion times are rewarded, with a maximum flight time of 180 seconds to complete the mission. Points for flight time are awarded as follows:

$`
\text{time points} = 10 \times \left(1 - \frac{\text{participant\_time} - \text{min\_time}}{\text{max\_time} - \text{min\_time}}\right)
`$

where:
- **participant_time**: Time taken by the participant to complete the task
- **min_time**: Minimum allowed time (10 seconds)
- **max_time**: Maximum allowed time (180 seconds)


## UAV Control and Available Topics
- **Challenge Start**: On the topic `/avader/challenge_start`, a True variable of DataType Bool will be published when the challenge starts, indicating that the participant can control the UAV.

- **Drone Control**: Participants control the UAV by publishing global positions to the `/avader/trajectory_setpoint` topic using the `px4_msgs/msg/TrajectorySetpoint` message type.

- **Drone Position**: Position of UAV can be listened on the topic `/fmu/out/vehicle_local_position`.

- **Camera**: The camera feed is available on the `/camera` topic, and camera parameters are accessible on the `/camera_info` topic. `<pose>0.12 0 -0.242 0 1.57 0</pose>` specify the camera's positional offset from the drone in the local NED frame.

- **Objects counting**: Localization of each individials should be published on `/avader/people_locations` topic using the `geometry_msgs/Pose` in the local NED. Location of every person should be published only once, in other case points will not be scored.
---

> **Important**  
> **DO NOT edit avader package.** You should only edit solver package.


This repository contains a Docker environment with ROS 2 Humble, Gazebo Harmonic, and PX4, providing an out-of-the-box setup for UAV simulations.

## Requirements

- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#container-device-interface-cdi-support)
- [VS Code Dev Containers Plugin](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container)

> **Important**  
> The recommended operating system is Ubuntu for full GUI support. Running on Windows requires a different configuration approach.

## Linux / Ubuntu
For Linux/Ubuntu systems, use the `.devcontainer` folder, which contains 4 `.yaml` files that differ by version.

- [compose.yaml] - configuration for GPU and building the image from the `Dockerfile` - [Initial Setup](#initial-setup)
- [compose.nogpu.yaml] - configuration for CPU and building the image from the `Dockerfile` - [Initial Setup](#initial-setup)
- [compose.image.yaml] - configuration for GPU and running the image from the file - [Docker Image from File](#docker-image-from-file)
- [compose.image.nogpu.yaml] - configuration for CPU and running the image from the file - [Docker Image from File](#docker-image-from-file)

## Windows
For Windows systems, use the `.devcontainer_windows` folder, which contains 4 `.yaml` files that differ by version.
The `.devcontainer` folder should be deleted, and the `.devcontainer_windows` folder should be renamed to `.devcontainer`.

- [compose.yaml] - configuration for GPU and building the image from the `Dockerfile` - [Initial Setup](#initial-setup)
- [compose.nogpu.yaml] - configuration for CPU and building the image from the `Dockerfile` - [Initial Setup](#initial-setup)
- [compose.image.yaml] - configuration for GPU and running the image from the file - [Docker Image from File](#docker-image-from-file)
- [compose.image.nogpu.yaml] - configuration for CPU and running the image from the file - [Docker Image from File](#docker-image-from-file)

## Docker Image from File
1. Download the file from Google Drive: [Docker Image - Rescue Challenge](https://drive.google.com/file/d/1S4v68ag1vSMfRCMBJmM5XyBAJRgVHzlp/view?usp=sharing).
2. Run the following command in the terminal, which may take up to 20 minutes:
```bash
docker image load < rescue_challenge.tar
```
3. In the `devcontainer.json` file, select the appropriate `compose.image*.yaml` file for the `dockerComposeFile` argument.
4. In the lower left corner, click the blue icon with two arrows pointing toward each other.
5. Select **"Open Folder in Container..."** from the dropdown menu and wait for Docker to build the container. (This may take up to 10 minutes with slower internet connections.)
6. Run the following commands:
```bash
cd ~/ros2_ws/
sudo ./setup.sh
```

## Initial Setup

1. Open VS Code in the project directory.
2. In the lower left corner, click the blue icon with two arrows pointing toward each other.
3. Select **"Open Folder in Container..."** from the dropdown menu and wait for Docker to build the container. (This may take up to 10 minutes with slower internet connections.)

> **Tip**  
> For Windows users with WSL 2, use the `Dockerfile.windows` and `compose.windows.yaml` files.

After the container builds, run the following commands:

```bash
cd ~/ros2_ws/
sudo ./setup.sh
./build.sh
source install/setup.bash
```

Before first start of the simulation you have tu execute below command. It's needed only before first start.
```bash
cd ~/PX4-Autopilot/ && make px4_sitl
```

## Run the challenge:
```bash
ros2 launch avader x500.launch.py
```

## Troubleshooting

### Gazebo GUI is not appearing
Terminal correctly starts and launches PX4 but **Gazebo GUI is not showing**:
```bash
ps aux | grep gz
```
If there is more than one develop+ task then you have to kill it manually:
```bash
kill -9 <PID>
```
```<PID>``` - you have to replace with the number next to the develop+, it is second parameter from left eg. 9378.

### Docker does not build

This docker requires nvidia graphics card so if you do not have, you have to edit .devcontainer/compose.yaml file. You have to remove following line:
```bash
runtime: nvidia
```
