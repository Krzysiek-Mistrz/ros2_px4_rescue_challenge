#!/bin/bash
set -e

cd /home/developer/ros2_ws

vcs import --recursive < /home/developer/ros2_ws/src/px4.repos src

rosdep update --rosdistro $ROS_DISTRO
rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y

sudo cp /home/developer/ros2_ws/src/.resources/lake_boats.sdf /home/developer/PX4-Autopilot/Tools/simulation/gz/worlds/
sudo cp /home/developer/ros2_ws/src/.resources/CMakeLists.txt /home/developer/PX4-Autopilot/src/modules/simulation/gz_bridge/
sudo cp -r /home/developer/ros2_ws/src/.resources/models/. /home/developer/PX4-Autopilot/Tools/simulation/gz/models/
