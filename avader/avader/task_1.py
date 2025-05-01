import rclpy
from rclpy.node import Node
from avader.topics import TopicsNode
from math import sqrt

class VehicleLocalPosition:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Destination(Node):
    def __init__(self):
        super().__init__('destination_node')
        self.topics = TopicsNode(self)

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.locations_to_visit = [
            (13, 20, -9),
            (-19.5, 10, -8.6),
            (16, -18.5, -13),
            (-20, -20, -11),
            (2, 3, -5.5)
        ]
        self.visited_flags = [False] * len(self.locations_to_visit)

    def timer_callback(self):
        uav_position = self.topics.get_uav_position()
        if uav_position is not None:
            for i, location in enumerate(self.locations_to_visit):
                if not self.visited_flags[i] and self.is_within_distance(uav_position, location, 0.4):
                    self.visited_flags[i] = True
        points = 40 / len(self.locations_to_visit) * sum(self.visited_flags)
        self.topics.points_1_publish(points)
        self.topics.publish_locations_to_visit(self.locations_to_visit)

    def is_within_distance(self, uav_position, location, threshold):
        distance = sqrt(
            (uav_position.x - location[0]) ** 2 +
            (uav_position.y - location[1]) ** 2 +
            (uav_position.z - location[2]) ** 2
        )
        return distance <= threshold

def main(args=None):
    rclpy.init(args=args)
    destination = Destination()
    rclpy.spin(destination)
    destination.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()