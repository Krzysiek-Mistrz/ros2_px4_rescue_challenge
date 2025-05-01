import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, Pose
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from std_msgs.msg import Int32, Float32
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy



class TopicsNode:
    def __init__(self, node: Node):
        self.node = node

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Variables to store the last messages
        self._last_camera_msg = None
        self._last_challenge_msg = None
        self._last_uav_control_msg = None
        self._last_people_count_msg = None
        self._last_task_1_msg = None
        self._last_task_2_msg = None
        self._last_task_3_msg = None
        self._uav_position = None
        self._last_people_loc = None
        
        # Subscriptions
        self.camera_subscription = self.node.create_subscription( Image, '/camera', self.camera_callback, 10)
        self.challenge_subscription = self.node.create_subscription(Bool, '/avader/challenge_start', self.challenge_callback, 10)
        self.uav_control_subscription = self.node.create_subscription(TrajectorySetpoint, '/avader/trajectory_setpoint', self.uav_control_callback, 10)
        self.people_count_subscription = self.node.create_subscription(Int32, '/avader/people_count', self.people_count_callback, 10)
        self.uav_position = self.node.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_position_callback, qos_profile)
        self.people_loc_subscription = self.node.create_subscription(Pose, '/avader/people_locations', self.people_loc_callback, 10)

        self.task_1_subscription = self.node.create_subscription(Float32, '/avader/_task_1_points', self.task_1_callback, 10)
        self.task_2_subscription = self.node.create_subscription(Float32, '/avader/_task_2_points', self.task_2_callback, 10)
        self.task_3_subscription = self.node.create_subscription(Float32, '/avader/_task_3_points', self.task_3_callback, 10)
        
        # Publishers
        self.camera_publisher = self.node.create_publisher(Image, '/avader/camera', 10)
        self.challenge_publisher = self.node.create_publisher(Bool, '/avader/challenge_start', 10)
        self.locations_publisher = self.node.create_publisher(PoseArray, '/avader/locations_to_visit', 10)

        self.points_1_publisher = self.node.create_publisher(Float32, '/avader/_task_1_points', 10)
        self.points_2_publisher = self.node.create_publisher(Float32, '/avader/_task_2_points', 10)
        self.points_3_publisher = self.node.create_publisher(Float32, '/avader/_task_3_points', 10)

        self.bridge = CvBridge()


    """
    Callbacks
    """
    def camera_callback(self, msg):
        self._last_camera_msg = msg


    def challenge_callback(self, msg):
        self._last_challenge_msg = msg


    def uav_control_callback(self, msg):
        self._last_uav_control_msg = msg


    def people_count_callback(self, msg):
        self._last_people_count_msg = msg


    def task_1_callback(self, msg):
        self._last_task_1_msg = msg

    
    def task_2_callback(self, msg):
        self._last_task_2_msg = msg

    
    def task_3_callback(self, msg):
        self._last_task_3_msg = msg

    
    def vehicle_position_callback(self, msg):
        self._uav_position = msg

    
    def people_loc_callback(self, msg):
        self._last_people_loc = msg


    """
    Subscribers
    """     
    def get_camera(self):
        return self._last_camera_msg


    def get_challenge_start(self):
        return self._last_challenge_msg


    def get_uav_control(self):
        return self._last_uav_control_msg
    

    def get_people_count(self):
        return self._last_people_count_msg
    

    def get_task_1(self):
        return self._last_task_1_msg
    

    def get_task_2(self):
        return self._last_task_2_msg
    

    def get_task_3(self):
        return self._last_task_3_msg


    def get_uav_position(self):
        return self._uav_position
    

    def get_people_loc(self):
        return self._last_people_loc


    """
    Publishers
    """
    def publish_camera(self, msg):
        self.camera_publisher.publish(msg)


    def publish_challenge(self, msg):
        self.challenge_publisher.publish(msg)


    def publish_locations_to_visit(self, msg):
        pose_array = PoseArray()
        pose_array.header.stamp = self.node.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'

        for loc in msg:
            pose = Pose()
            pose.position.x = float(loc[0])
            pose.position.y = float(loc[1])
            pose.position.z = float(loc[2])
            pose_array.poses.append(pose)

        self.locations_publisher.publish(pose_array)


    def points_1_publish(self, msg):
        var = Float32()
        var.data = msg
        self.points_1_publisher.publish(var)


    def points_2_publish(self, msg):
        var = Float32()
        var.data = msg
        self.points_2_publisher.publish(var)


    def points_3_publish(self, msg):
        var = Float32()
        var.data = float(msg)
        self.points_3_publisher.publish(var)