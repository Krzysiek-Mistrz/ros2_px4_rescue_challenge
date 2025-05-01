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

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class SolverClass(Node):
    def __init__(self):
        super().__init__('solver_node')

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
        self.target_points = []  # lista krotek
        self.detected_objects_positions = []  # lista krotek
        self.mission_started = False
        self.uav_position = None  # załóżmy że krotka
        self.bridge = CvBridge()
        self.min_dist = 0.2  # Minimalny akceptowalny dystans między położeniem obiektu z bazy a nowo wykrytym punktem
        self.path_to_take = None
        self.currnet_target_point = None
        self.temp = True
        self.number_of_all_detections = 0

        # przerwania od timera
        # self.timer = self.create_timer(1.0 / 30.0, self.main)
        # 
        self.timer2 = self.create_timer(1, self.navigate_uav)

    def start_callback(self, msg):
        if msg.data:
            self.mission_started = True
            #debug
            #self.get_logger().info("Mission started.")

    def locations_callback(self, msg):
        self.target_points = [(pose.position.x, pose.position.y, pose.position.z) for pose in msg.poses]
        #debug
        #self.get_logger().info(f"Received target locations: {self.target_points}")
        if self.path_to_take is None:
            self.plan_trajectory()


    def uav_position_callback(self, msg):
        try:
            # Upewnij się, że wszystkie atrybuty są obecne
            self.uav_position = (msg.x, msg.y, msg.z)
            #debug
            #self.get_logger().info(f"UAV position updated: {self.uav_position}")
        except AttributeError as e:
            self.get_logger().error(f"Invalid VehicleLocalPosition message: {e}")


    # chyba nigdzie nieużywna metoda
    def decode_image(self, image_msg):
        # Decode ROS image to OpenCV format
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame


    def camera_callback(self, msg):
        # Konwersja obrazu ROS do formatu OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is None or frame.size == 0:
            self.get_logger().error("Received empty frame from camera.")
            return

        # Wykrywanie obiektów
        detected_positions = self.detect_objects(frame)
        center_of_frame = (frame.shape[1] // 2, frame.shape[0] // 2)  # x, y

        for px, py in detected_positions:
            self.check_and_add_to_detected(px, py, center_of_frame)


    def detect_objects(self, frame, threshold=200):
        if frame is None or frame.size == 0:
            self.get_logger().error("Frame is empty, skipping detection.")
            return []

        # Konwersja do przestrzeni kolorów RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Definicja progów dla każdego kanału
        channel1_min, channel1_max = 86, 255
        channel2_min, channel2_max = 0, 75
        channel3_min, channel3_max = 0, 94

        # Utworzenie maski
        mask = (
            (image_rgb[:, :, 0] >= channel1_min) & (image_rgb[:, :, 0] <= channel1_max) &
            (image_rgb[:, :, 1] >= channel2_min) & (image_rgb[:, :, 1] <= channel2_max) &
            (image_rgb[:, :, 2] >= channel3_min) & (image_rgb[:, :, 2] <= channel3_max)
        )
        binary_mask = np.zeros_like(image_rgb[:, :, 0], dtype=np.uint8)
        binary_mask[mask] = 255

        # Znajdowanie konturów
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_positions = []

        for contour in contours:
            if cv2.contourArea(contour) > 900:  # Filtrowanie małych obszarów
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_positions.append((cx, cy))

        return detected_positions

        """# Zlicz białe piksele
        white_pixel_count = np.sum(binary_mask == 255)
        if white_pixel_count > threshold:
            self.number_of_all_detections += 1"""

        """contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        object_positions = []
        center_of_frame = ( frame.shape[0] //2, frame.shape[1] //2) # framey chyba będą tego samego rozmiaru więc można policzyć i na stałe
        
        for contour in contours:
            if cv2.contourArea(contour) > 900:  # Minimum area to filter noise  # tutaj trzeba będzie zmieniać wielkość minimalnej wartości                                   # jeśli uav będzie zmieniał wysokość lotu
                self.number_of_all_detections += 1"""
    

    def check_and_add_to_detected(self, px: float, py: float, center_of_frame):
        '''Funkcja licząca położenie zidentyfikowanego obiektu i dodająca go do bazy, jeśli nie jest duplikatem.'''

        # Współczynniki (mogą być parametryzowane, jeśli są stałe dla różnych wysokości)
        a_hor = -4.163469105855614
        b_hor = 75.2155798055519
        a_ver = -6.245217604480979
        b_ver = 112.8236216457635

        uav_x, uav_y, h = self.uav_position
        h = -h  # Wysokość UAV (ujemna w zależności od układu współrzędnych)

        # Przeliczniki z pikseli na metry
        meters_to_pixels_horizontal = a_hor * h + b_hor
        meters_to_pixels_vertical = a_ver * h + b_ver
        pixels_to_meters_horizontal = 1 / meters_to_pixels_horizontal
        pixels_to_meters_vertical = 1 / meters_to_pixels_vertical

        # Wyliczenie rzeczywistej pozycji obiektu
        obj_x = uav_x + pixels_to_meters_horizontal * (px - center_of_frame[0])
        obj_y = uav_y + pixels_to_meters_vertical * (py - center_of_frame[1])

        # Sprawdzenie, czy obiekt już istnieje w bazie
        for prev_x, prev_y, _ in self.detected_objects_positions:
            dist = sqrt((obj_x - prev_x) ** 2 + (obj_y - prev_y) ** 2)
            if dist < self.min_dist:  # Obiekt już istnieje w bazie
                return

        # Dodanie nowego obiektu
        self.detected_objects_positions.append((obj_x, obj_y, 0))


    def publish_people_count(self, count):
        msg = Int32()
        msg.data = count
        self.people_count_publisher.publish(msg)

    # tu nie rozumiem clocka ale chodzi o wstawienie loializacji obiektów
    def report_object_locations(self, positions):
        # Convert detected object positions to PoseArray and publish
        self.temp = False
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'

        for position in positions:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(position[0]), float(position[1]), 0.0
            pose_array.poses.append(pose)
            self.get_logger().info(f"DANO POZYCJE!!!: {(float(position[0]), float(position[1]), 0.0)}")

        self.object_locations_publisher.publish(pose_array)


    def plan_trajectory(self):
        if self.uav_position is None:
            self.get_logger().warn("UAV position is None")
            return

        # Plan an optimal trajectory using TSP
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

        # poprzestawianie wierzchołków w cyklu tak żeby na start lecieć bezpośrednio z self.uav_position do odpowiedniego wierzchołka
        cycle = [points[i] for i in path]   # cykl położeniami punktów
        start_index = cycle.index(starting_point)
        final_path = cycle[(start_index + 1):-1]    # ostetczna kolejność odwiedzania wirzchołków wylatując z starting_point
        for i in cycle[:start_index]:
            final_path.append(i)

        # asign complete trajectory
        self.path_to_take = final_path.copy()

    # bardzo zmienione
    def publish_trajectory(self):
        point = self.currnet_target_point
        msg = TrajectorySetpoint()
        msg.velocity = [0.1, 0.1, 0.1]
        msg.position = [point[0], point[1], point[2]]
        self.trajectory_publisher.publish(msg)

    # nawigacja plus wysłanie ilości obieków i ich lokalizacji pod koniec "misji"
    def navigate_uav(self):
        # start flight towards firs target
        if self.mission_started and self.path_to_take is not None and self.currnet_target_point is None:
            self.currnet_target_point = self.path_to_take.pop(0)
            self.publish_trajectory()

        if self.mission_started and self.path_to_take is not None:
            ct_x, ct_y, ct_z = self.currnet_target_point # currnet target coordinates
            uav_x, uav_y, uav_z = self.uav_position # current uav position

            # calcualete sidtance
            dist = sqrt((ct_x - uav_x) ** 2 + (ct_y - uav_y) ** 2 + (ct_z - uav_z) ** 2)

            # go to next point in path_to_take if close to last point publish people counta and their locations
            if dist < self.min_dist:
                try:
                    self.currnet_target_point = self.path_to_take.pop(0)
                    self.publish_trajectory()
                except IndexError:
                    if self.temp:
                        self.publish_people_count(self.number_of_all_detections)
                        self.report_object_locations(self.detected_objects_positions)


def main(args=None):
    rclpy.init(args=args)
    node = SolverClass()
    rclpy.spin(node)
    # node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()