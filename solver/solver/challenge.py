from rclpy.node import Node
import rclpy
from std_msgs.msg import Bool, Int32
from px4_msgs.msg import TrajectorySetpoint
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
import cv2
import numpy as np
from math import sqrt
import networkx as nx
from cv_bridge import CvBridge
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class SolverClass(Node):
    def __init__(self):
        super().__init__('solver_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # Publishers
        self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, '/avader/trajectory_setpoint', 10)
        self.people_count_publisher = self.create_publisher(Int32, '/avader/people_count', 10)
        self.object_locations_publisher = self.create_publisher(PoseArray, '/avader/people_locations', 10)
        # Subscribers
        self.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position', 
            self.uav_position_callback, 
            qos_profile
        )
        self.create_subscription(
            PoseArray, 
            '/avader/locations_to_visit', 
            self.locations_callback, 
            10
        )
        self.create_subscription(
            Image, 
            '/camera', 
            self.camera_callback, 
            10
        )
        self.create_subscription(
            Bool, 
            '/avader/challenge_start', 
            self.start_callback, 
            10
        )
        # State Variables
        self.target_points = []
        self.detected_objects_positions = []
        self.mission_started = False
        self.uav_position = None
        self.bridge = CvBridge()
        self.min_dist = 0.2
        self.path_to_take = None
        self.currnet_target_point = None
        self.mission_completed = False
        self.number_of_all_detections = 0
        self.reported_results = False
        self.timer2 = self.create_timer(1, self.navigate_uav)

    def start_callback(self, msg):
        if msg.data:
            self.mission_started = True
            self.mission_completed = False
            self.reported_results = False
            self.detected_objects_positions = []
            self.number_of_all_detections = 0
            self.get_logger().info("Mission started.")

    def locations_callback(self, msg):
        self.target_points = [(pose.position.x, pose.position.y, pose.position.z) for pose in msg.poses]
        if self.path_to_take is None:
            self.plan_trajectory()

    def uav_position_callback(self, msg):
        try:
            self.uav_position = (msg.x, msg.y, msg.z)
        except AttributeError as e:
            self.get_logger().error(f"Invalid VehicleLocalPosition message: {e}")

    def camera_callback(self, msg):
        if not self.mission_started or self.mission_completed:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is None or frame.size == 0:
            self.get_logger().error("Received empty frame from camera.")
            return
        self.detect_objects(frame)

    def detect_objects(self, frame, threshold=200):
        if frame is None or frame.size == 0:
            self.get_logger().error("Frame is empty, skipping detection.")
            return []
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        channel1_min, channel1_max = 86, 255
        channel2_min, channel2_max = 0, 75
        channel3_min, channel3_max = 0, 94
        mask = (
            (image_rgb[:, :, 0] >= channel1_min) & (image_rgb[:, :, 0] <= channel1_max) &
            (image_rgb[:, :, 1] >= channel2_min) & (image_rgb[:, :, 1] <= channel2_max) &
            (image_rgb[:, :, 2] >= channel3_min) & (image_rgb[:, :, 2] <= channel3_max)
        )
        binary_mask = np.zeros_like(image_rgb[:, :, 0], dtype=np.uint8)
        binary_mask[mask] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 900:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_positions.append((cx, cy))
        center_of_frame = (frame.shape[1] // 2, frame.shape[0] // 2)
        for px, py in detected_positions:
            self.check_and_add_to_detected(px, py, center_of_frame)
        return detected_positions    

    def check_and_add_to_detected(self, px: float, py: float, center_of_frame):
        '''Funkcja licząca położenie zidentyfikowanego obiektu i dodająca go do bazy, jeśli nie jest duplikatem.'''
        if self.uav_position is None:
            return
        a_hor = -4.163469105855614
        b_hor = 75.2155798055519
        a_ver = -6.245217604480979
        b_ver = 112.8236216457635
        uav_x, uav_y, h = self.uav_position
        h = -h
        meters_to_pixels_horizontal = a_hor * h + b_hor
        meters_to_pixels_vertical = a_ver * h + b_ver
        pixels_to_meters_horizontal = 1 / meters_to_pixels_horizontal
        pixels_to_meters_vertical = 1 / meters_to_pixels_vertical
        obj_x = uav_x + pixels_to_meters_horizontal * (px - center_of_frame[0])
        obj_y = uav_y + pixels_to_meters_vertical * (py - center_of_frame[1])
        for prev_x, prev_y, _ in self.detected_objects_positions:
            dist = sqrt((obj_x - prev_x) ** 2 + (obj_y - prev_y) ** 2)
            if dist < self.min_dist:
                return
        self.detected_objects_positions.append((obj_x, obj_y, 0))
        self.number_of_all_detections += 1
        self.get_logger().info(f"New object detected at ({obj_x:.2f}, {obj_y:.2f}). Total: {self.number_of_all_detections}")

    def publish_people_count(self, count):
        msg = Int32()
        msg.data = count
        self.people_count_publisher.publish(msg)

    def report_object_locations(self, positions):
        if not positions:
            self.get_logger().warn("No objects detected to report")
            return            
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        for position in positions:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(position[0]), float(position[1]), 0.0
            pose_array.poses.append(pose)
            self.get_logger().info(f"Reporting position: {(float(position[0]), float(position[1]), 0.0)}")
        self.object_locations_publisher.publish(pose_array)
        self.reported_results = True

    def plan_trajectory(self):
        if self.uav_position is None:
            self.get_logger().warn("UAV position is None")
            return
        G = nx.complete_graph(len(self.target_points) + 1)
        points = self.target_points.copy()
        starting_point = self.uav_position
        points.append(starting_point)
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points):
                if i != j:
                    distance = sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
                    G[i][j]['weight'] = distance
        path = nx.approximation.traveling_salesman_problem(G)
        cycle = [points[i] for i in path]
        start_index = cycle.index(starting_point)
        final_path = cycle[(start_index + 1):-1]
        for i in cycle[:start_index]:
            final_path.append(i)
        self.path_to_take = final_path.copy()

    def publish_trajectory(self):
        point = self.currnet_target_point
        msg = TrajectorySetpoint()
        msg.velocity = [0.1, 0.1, 0.1]
        msg.position = [point[0], point[1], point[2]]
        self.trajectory_publisher.publish(msg)

    def navigate_uav(self):
        if not self.mission_started or self.uav_position is None:
            return
        if self.mission_completed and self.reported_results:
            return
        if self.path_to_take is not None and self.currnet_target_point is None:
            self.currnet_target_point = self.path_to_take.pop(0)
            self.publish_trajectory()
            self.get_logger().info(f"Starting navigation to first point: {self.currnet_target_point}")
        if self.path_to_take is not None and self.currnet_target_point is not None:
            ct_x, ct_y, ct_z = self.currnet_target_point
            uav_x, uav_y, uav_z = self.uav_position
            dist = sqrt((ct_x - uav_x) ** 2 + (ct_y - uav_y) ** 2 + (ct_z - uav_z) ** 2)
            if dist < self.min_dist:
                try:
                    self.currnet_target_point = self.path_to_take.pop(0)
                    self.publish_trajectory()
                    self.get_logger().info(f"Moving to next point: {self.currnet_target_point}")
                except IndexError:
                    if not self.mission_completed:
                        self.mission_completed = True
                        self.get_logger().info(f"Mission completed. Detected {self.number_of_all_detections} objects.")
                        self.publish_people_count(self.number_of_all_detections)
                        self.report_object_locations(self.detected_objects_positions)


def main(args=None):
    rclpy.init(args=args)
    node = SolverClass()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
