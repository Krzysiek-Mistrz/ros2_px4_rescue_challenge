from avader.topics import TopicsNode
from rclpy.node import Node
import rclpy
import math

class PeopleLoc(Node):
    def __init__(self):
        super().__init__('people_location')
        
        self.topics = TopicsNode(self)
        self.timer = self.create_timer(1.0 / 60.0, self.main)

        self.people_positions = [
            (3, -5),
            (-10, 15),
            (20, -10),
            (-22, -20),
            (15, 20),
            (-18, 5),
            (10, -18),
            (-7, -10)
        ]
        
        self.status_list = [False] * len(self.people_positions)
        
    def main(self):
        loc = self.topics.get_people_loc()
        
        if loc:
            self.check_location(loc)
            points = self.calculate_points(30)
            self.topics.points_3_publish(points)

    def check_location(self, loc):
        for i, (x, y) in enumerate(self.people_positions):
            distance = math.sqrt((loc.position.x - x) ** 2 + (loc.position.y - y) ** 2)
            if distance <= 0.8:
                self.status_list[i] = True

    def calculate_points(self, max_points):
        num_correct = sum(self.status_list)
        if num_correct > 0:
            points = max_points / len(self.people_positions) * num_correct
        else:
            points = 0
        return points


def main(args=None):
    rclpy.init(args=args)
    people_location = PeopleLoc()
    rclpy.spin(people_location)
    people_location.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
