from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    challenge_node = Node(
        package='solver',
        executable='challenge',
        name='challenge',
        output='screen'
    )

    return LaunchDescription([
        challenge_node
    ])
