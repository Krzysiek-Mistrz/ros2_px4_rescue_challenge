import rclpy
from rclpy.node import Node
from avader.topics import TopicsNode

# TASK 2
# This task requires you to count the number of people.

class CountPeople(Node):
    def __init__(self):
        super().__init__('count_people_node')
        self.topics = TopicsNode(self)
        self.data_send = False

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        if self.topics.get_people_count() is not None and self.data_send is False:
            self.data_send = True
            count = self.topics.get_people_count().data
            if count == 8:
                self.topics.points_2_publish(20.0)
            else:
                self.topics.points_2_publish(0.0)


def main(args=None):
    rclpy.init(args=args)
    count_people_node = CountPeople()
    rclpy.spin(count_people_node)
    count_people_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()