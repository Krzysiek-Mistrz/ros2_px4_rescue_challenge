import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import datetime
from avader.topics import TopicsNode


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.MAX_TIME = 180
        self.MIN_TIME = 10
        self.all_points = 0.0
        self.topics = TopicsNode(self)
        self.timer = self.create_timer(1.0 / 30.0, self.main_loop)

    def main_loop(self):
        frame = self.topics.get_camera()
        if frame is None:
            return
        
        frame = self.convert_image(frame, 'cv2')
        frame = self.draw_time(frame)
        frame = self.draw_points(frame)
        
        frame = self.convert_image(frame, 'ros')

        self.topics.publish_camera(frame)
    
    def draw_points(self, frame):
        if hasattr(self, 'start_time'):
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
            if elapsed_time > self.MAX_TIME:
                font = cv2.FONT_HERSHEY_SIMPLEX
                final_score_text = "FINAL SCORE"
                text_size = cv2.getTextSize(final_score_text, font, 1.5, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] - text_size[1]) // 2
                cv2.putText(frame, final_score_text, (text_x, text_y), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, final_score_text, (text_x, text_y), font, 1.5, (139, 0, 0), 2, cv2.LINE_AA)

                points_text = f"{self.all_points:.2f}"
                text_size = cv2.getTextSize(points_text, font, 1.2, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2 + 40
                cv2.putText(frame, points_text, (text_x, text_y), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, points_text, (text_x, text_y), font, 1.2, (139, 0, 0), 2, cv2.LINE_AA)
                return frame
            
        self.all_points = 0.0

        # TASK 1
        points = self.topics.get_task_1()
        if points is not None:
            if points.data !=0:
                self.all_points += points.data
                frame = cv2.putText(frame, f"Task 1: {points.data if points is not None else 0.0}/40", (20, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, f"Task 1: {points.data if points is not None else 0.0}/40", (20, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, "Task 1 not completed", (20, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # TASK 2
        points = self.topics.get_task_2()
        if points is not None:
            if points.data != 0:
                self.all_points += points.data
                frame = cv2.putText(frame, f"Task 2: {points.data if points is not None else 0.0}/20", (20, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, f"Task 2: {points.data if points is not None else 0.0}/20", (20, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, "Task 2 not completed", (20, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # TASK 3
        points = self.topics.get_task_3()
        if points is not None:
            if points.data !=0:
                self.all_points += points.data
                frame = cv2.putText(frame, f"Task 3: {points.data if points is not None else 0.0}/30", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, f"Task 3: {points.data if points is not None else 0.0}/30", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, "Task 3 not completed", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        self.all_points += self.time_points()

        

        font = cv2.FONT_HERSHEY_SIMPLEX
        points_text = f"Points: {self.all_points:.2f}"
        text_size = cv2.getTextSize(points_text, font, 0.7, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 40
        cv2.putText(frame, points_text, (text_x, text_y), font, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, points_text, (text_x, text_y), font, 0.7, (139, 0, 0), 2, cv2.LINE_AA)
        return frame


    def time_points(self):
        if not hasattr(self, 'start_time'):
            return 0.0
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        points = 10 * (1 - ((max(min(elapsed_time, self.MAX_TIME), self.MIN_TIME) - self.MIN_TIME)/(self.MAX_TIME - self.MIN_TIME)))

        return points
    
    def draw_time(self, image):
        if self.topics.get_challenge_start() is not None:
            if self.topics.get_challenge_start().data:
                if not hasattr(self, 'start_time'):
                    self.start_time = datetime.datetime.now()
                font = cv2.FONT_HERSHEY_SIMPLEX
                elapsed_time = datetime.datetime.now() - self.start_time
                elapsed_seconds = int(elapsed_time.total_seconds())
                minutes, seconds = divmod(elapsed_seconds, 60)
                milliseconds = int((elapsed_time.total_seconds() - elapsed_seconds) * 1000)
                timer_text = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
                text_size = cv2.getTextSize(timer_text, font, 0.7, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] - 10
                cv2.putText(image, timer_text, (text_x, text_y), font, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(image, timer_text, (text_x, text_y), font, 0.7, (139, 0, 0), 2, cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "Waiting for start..."
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] - 10
                cv2.putText(image, text, (text_x, text_y), font, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(image, text, (text_x, text_y), font, 0.7, (139, 0, 0), 2, cv2.LINE_AA)
        return image

    def convert_image(self, image, encoding):
        if encoding == 'cv2':
            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        elif encoding == 'ros':
            bridge = CvBridge()
            image = bridge.cv2_to_imgmsg(image, encoding='bgr8')
        return image


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    while rclpy.ok():
        rclpy.spin_once(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()