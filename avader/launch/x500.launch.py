import os
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch import LaunchDescription
# from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler
# from launch.events.process import ProcessStarted
from launch.event_handlers.on_process_start import OnProcessStart

from ament_index_python.packages import get_package_share_directory as FindPackageShare

def generate_launch_description():
    env = {
        'GZ_SIM_RESOURCE_PATH': '/home/developer/PX4-Autopilot/Tools/simulation/gz/worlds',
        'PX4_SYS_AUTOSTART': '4010',
        'PX4_GZ_WORLD': 'lake_boats',
        'PX4_GZ_MODEL_POSE': '0,0,1',
        'PX4_GZ_MODEL': 'gz_x500_mono_cam'
    }

    px4_sitl = ExecuteProcess(
        cmd=['/home/developer/PX4-Autopilot/build/px4_sitl_default/bin/px4'],
        # cwd=os.path.expanduser('~/PX4-Autopilot'),
        output='screen',
        additional_env=env
    )

    micro_ros_agent = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['udp4', '--port', '8888'],
        output='screen'
    )

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
        ],
        output='screen'
    )
    
    hover_start = Node(
        package='avader',
        executable='hover_start',
        name='hover_start',
        output='screen'
    )

    avader_camera = Node(
        package='avader',
        executable='camera',
        name='camera',
        output='screen'
    )

    task_1 = Node(
        package='avader',
        executable='task_1',
        name='task_1',
        output='screen'
    )

    task_2 = Node(
        package='avader',
        executable='task_2',
        name='task_2',
        output='screen'
    )

    task_3 = Node(
        package='avader',
        executable='task_3',
        name='task_3',
        output='screen'
    )

    tasks_solver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('solver'),
            '/launch/tasks_solver.launch.py'
        ])
    )

    return LaunchDescription([
        micro_ros_agent,
        RegisterEventHandler(
            OnProcessStart(
                target_action=micro_ros_agent,
                on_start=[px4_sitl]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=px4_sitl,
                on_start=[camera_bridge]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=camera_bridge,
                on_start=[hover_start, avader_camera, task_1, task_2, task_3, tasks_solver_launch]
            )
        ),
    ])
