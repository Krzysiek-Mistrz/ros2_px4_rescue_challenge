# Bridge communication between ROS and Gazebo

This package provides a network bridge which enables the exchange of messages
between ROS and Gazebo Transport.

The following message types can be bridged for topics:

| ROS type                                       | Gazebo Transport Type               |
| ---------------------------------------------- | :------------------------------:    |
| actuator_msgs/msg/Actuators                    | gz.msgs.Actuators                   |
| builtin_interfaces/msg/Time                    | gz.msgs.Time                        |
| geometry_msgs/msg/Point                        | gz.msgs.Vector3d                    |
| geometry_msgs/msg/Pose                         | gz.msgs.Pose                        |
| geometry_msgs/msg/PoseArray                    | gz.msgs.Pose_V                      |
| geometry_msgs/msg/PoseStamped                  | gz.msgs.Pose                        |
| geometry_msgs/msg/PoseWithCovariance           | gz.msgs.PoseWithCovariance          |
| geometry_msgs/msg/PoseWithCovarianceStamped    | gz.msgs.PoseWithCovariance          |
| geometry_msgs/msg/Quaternion                   | gz.msgs.Quaternion                  |
| geometry_msgs/msg/Transform                    | gz.msgs.Pose                        |
| geometry_msgs/msg/TransformStamped             | gz.msgs.Pose                        |
| geometry_msgs/msg/Twist                        | gz.msgs.Twist                       |
| geometry_msgs/msg/TwistStamped                 | gz.msgs.Twist                       |
| geometry_msgs/msg/TwistWithCovariance          | gz.msgs.TwistWithCovariance         |
| geometry_msgs/msg/TwistWithCovarianceStamped   | gz.msgs.TwistWithCovariance         |
| geometry_msgs/msg/Vector3                      | gz.msgs.Vector3d                    |
| geometry_msgs/msg/Wrench                       | gz.msgs.Wrench                      |
| geometry_msgs/msg/WrenchStamped                | gz.msgs.Wrench                      |
| gps_msgs/msg/GPSFix                            | gz.msgs.NavSat                      |
| nav_msgs/msg/Odometry                          | gz.msgs.Odometry                    |
| nav_msgs/msg/Odometry                          | gz.msgs.OdometryWithCovariance      |
| rcl_interfaces/msg/ParameterValue              | gz.msgs.Any                         |
| ros_gz_interfaces/msg/Altimeter                | gz.msgs.Altimeter                   |
| ros_gz_interfaces/msg/Contact                  | gz.msgs.Contact                     |
| ros_gz_interfaces/msg/Contacts                 | gz.msgs.Contacts                    |
| ros_gz_interfaces/msg/Dataframe                | gz.msgs.Dataframe                   |
| ros_gz_interfaces/msg/Entity                   | gz.msgs.Entity                      |
| ros_gz_interfaces/msg/EntityWrench             | gz.msgs.EntityWrench                |
| ros_gz_interfaces/msg/Float32Array             | gz.msgs.Float_V                     |
| ros_gz_interfaces/msg/GuiCamera                | gz.msgs.GUICamera                   |
| ros_gz_interfaces/msg/JointWrench              | gz.msgs.JointWrench                 |
| ros_gz_interfaces/msg/Light                    | gz.msgs.Light                       |
| ros_gz_interfaces/msg/ParamVec                 | gz.msgs.Param                       |
| ros_gz_interfaces/msg/ParamVec                 | gz.msgs.Param_V                     |
| ros_gz_interfaces/msg/SensorNoise              | gz.msgs.SensorNoise                 |
| ros_gz_interfaces/msg/StringVec                | gz.msgs.StringMsg_V                 |
| ros_gz_interfaces/msg/TrackVisual              | gz.msgs.TrackVisual                 |
| ros_gz_interfaces/msg/VideoRecord              | gz.msgs.VideoRecord                 |
| rosgraph_msgs/msg/Clock                        | gz.msgs.Clock                       |
| sensor_msgs/msg/BatteryState                   | gz.msgs.BatteryState                |
| sensor_msgs/msg/CameraInfo                     | gz.msgs.CameraInfo                  |
| sensor_msgs/msg/FluidPressure                  | gz.msgs.FluidPressure               |
| sensor_msgs/msg/Image                          | gz.msgs.Image                       |
| sensor_msgs/msg/Imu                            | gz.msgs.IMU                         |
| sensor_msgs/msg/JointState                     | gz.msgs.Model                       |
| sensor_msgs/msg/Joy                            | gz.msgs.Joy                         |
| sensor_msgs/msg/LaserScan                      | gz.msgs.LaserScan                   |
| sensor_msgs/msg/MagneticField                  | gz.msgs.Magnetometer                |
| sensor_msgs/msg/NavSatFix                      | gz.msgs.NavSat                      |
| sensor_msgs/msg/PointCloud2                    | gz.msgs.PointCloudPacked            |
| std_msgs/msg/Bool                              | gz.msgs.Boolean                     |
| std_msgs/msg/ColorRGBA                         | gz.msgs.Color                       |
| std_msgs/msg/Empty                             | gz.msgs.Empty                       |
| std_msgs/msg/Float32                           | gz.msgs.Float                       |
| std_msgs/msg/Float64                           | gz.msgs.Double                      |
| std_msgs/msg/Header                            | gz.msgs.Header                      |
| std_msgs/msg/Int32                             | gz.msgs.Int32                       |
| std_msgs/msg/String                            | gz.msgs.StringMsg                   |
| std_msgs/msg/UInt32                            | gz.msgs.UInt32                      |
| tf2_msgs/msg/TFMessage                         | gz.msgs.Pose_V                      |
| trajectory_msgs/msg/JointTrajectory            | gz.msgs.JointTrajectory             |
| vision_msgs/msg/Detection2D                    | gz.msgs.AnnotatedAxisAligned2DBox   |
| vision_msgs/msg/Detection2DArray               | gz.msgs.AnnotatedAxisAligned2DBox_V |
| vision_msgs/msg/Detection3D                    | gz::msgs::AnnotatedOriented3DBox    |
| vision_msgs/msg/Detection3DArray               | gz::msgs::AnnotatedOriented3DBox_V  |

## Launching the bridge manually

We can initialize a bidirectional bridge so we can have ROS as the publisher and Gazebo as the subscriber or vice versa. The syntax is `/TOPIC@ROS_MSG@GZ_MSG`, such that `TOPIC` is the Gazebo internal topic, `ROS_MSG` is the ROS message type for this topic, and `GZ_MSG` is the Gazebo message type.

```bash
ros2 run ros_gz_bridge parameter_bridge /TOPIC@ROS_MSG@GZ_MSG
```

For example:

```
ros2 run ros_gz_bridge parameter_bridge /scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan
```

The `ros2 run ros_gz_bridge parameter_bridge` command simply runs the `parameter_bridge` code from the `ros_gz_bridge` package. Then, we specify our topic `/TOPIC` over which the messages will be sent. The first `@` symbol delimits the topic name from the message types. Following the first `@` symbol is the ROS message type.

The ROS message type is followed by an `@`, `[`, or `]` symbol where:

* `@`  is a bidirectional bridge.
* `[`  is a bridge from Gazebo to ROS.
* `]`  is a bridge from ROS to Gazebo.

Have a look at these [examples]( https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md#example-1a-gazebo-transport-talker-and-ros-2-listener)
explaining how to make communication connections from ROS to Gazebo and vice versa.

It is also possible to use ROS Launch with the `ros_gz_bridge` and represent the topics in yaml format to be given to the bridge at launch time.

```yaml
ros_topic_name: "scan"
gz_topic_name: "/scan"
ros_type_name: "sensor_msgs/msg/LaserScan"
gz_type_name: "gz.msgs.LaserScan"
direction: GZ_TO_ROS  # BIDIRECTIONAL or ROS_TO_GZ
```

The configuration file is a YAML file that contains the mapping between the ROS
and Gazebo topics to be bridged. For each pair of topics to be bridged, the
following parameters are accepted:

* `ros_topic_name`: The topic name on the ROS side.
* `gz_topic_name`: The corresponding topic name on the Gazebo side.
* `ros_type_name`: The type of this ROS topic.
* `gz_type_name`: The type of this Gazebo topic.
* `subscriber_queue`: The size of the ROS subscriber queue.
* `publisher_queue`: The size of the ROS publisher queue.
* `lazy`: Whether there's a lazy subscriber or not. If there's no real
subscribers the bridge won't create the internal subscribers either. This should
speedup performance.
* `direction`: It's possible to specify `GZ_TO_ROS`, `ROS_TO_GZ` and
`BIDIRECTIONAL`.

## Configuring the Bridge via YAML

When configuring many topics, it is easier to use a file-based configuration in a markup
language. In this case, the `ros_gz` bridge supports using a YAML file to configure the
various parameters.

The configuration file must be a YAML array of maps.
An example configuration for 5 bridges is below, showing the various ways that a
bridge may be specified:

```yaml
 # Set just topic name, applies to both
- topic_name: "chatter"
  ros_type_name: "std_msgs/msg/String"
  gz_type_name: "gz.msgs.StringMsg"

# Set just ROS topic name, applies to both
- ros_topic_name: "chatter_ros"
  ros_type_name: "std_msgs/msg/String"
  gz_type_name: "gz.msgs.StringMsg"

# Set just GZ topic name, applies to both
- gz_topic_name: "chatter_gz"
  ros_type_name: "std_msgs/msg/String"
  gz_type_name: "gz.msgs.StringMsg"

# Set each topic name explicitly
- ros_topic_name: "chatter_both_ros"
  gz_topic_name: "chatter_both_gz"
  ros_type_name: "std_msgs/msg/String"
  gz_type_name: "gz.msgs.StringMsg"

# Full set of configurations
- ros_topic_name: "ros_chatter"
  gz_topic_name: "gz_chatter"
  ros_type_name: "std_msgs/msg/String"
  gz_type_name: "gz.msgs.StringMsg"
  subscriber_queue: 5       # Default 10
  publisher_queue: 6        # Default 10
  lazy: true                # Default "false"
  direction: BIDIRECTIONAL  # Default "BIDIRECTIONAL" - Bridge both directions
                            # "GZ_TO_ROS" - Bridge Gz topic to ROS
                            # "ROS_TO_GZ" - Bridge ROS topic to Gz
```

To run the bridge node with the above configuration:
```bash
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$WORKSPACE/ros_gz/ros_gz_bridge/test/config/full.yaml
```

## Run the bridge and exchange images

In this example, we're going to generate Gazebo Transport images using
Gazebo Sim, that will be converted into ROS images, and visualized with
`rqt_image_viewer`.

First we start Gazebo Sim (don't forget to hit play, or Gazebo Sim won't generate any images).

```bash
# Shell A:
gz sim sensors_demo.sdf
```

Let's see the topic where camera images are published.

```bash
# Shell B:
gz topic -l | grep image
/rgbd_camera/depth_image
/rgbd_camera/image
```

Then we start the parameter bridge with the previous topic.

```bash
# Shell B:
. ~/bridge_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge /rgbd_camera/image@sensor_msgs/msg/Image@gz.msgs.Image
```

Now we start the ROS GUI:

```bash
# Shell C:
. /opt/ros/rolling/setup.bash
ros2 run rqt_image_view rqt_image_view /rgbd_camera/image
```

You should see the current images in `rqt_image_view` which are coming from
Gazebo (published as Gazebo Msgs over Gazebo Transport).

The screenshot shows all the shell windows and their expected content
(it was taken using ROS 2 Galactic and Gazebo Fortress):

![Gazebo Transport images and ROS rqt](images/bridge_image_exchange.png)