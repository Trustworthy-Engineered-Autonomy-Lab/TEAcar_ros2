from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bringup_share = get_package_share_directory('bringup')
    nn_controller_share = get_package_share_directory('nn_controller')

    drive_launch = os.path.join(bringup_share, 'launch', 'drive.launch.py')
    nn_params = os.path.join(nn_controller_share, 'config', 'nn_controller.yaml')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(drive_launch)
        ),

        Node(
            package='controller',
            executable='nn_controller_node',
            name='nn_controller_node',
            output='screen',
            parameters=[nn_params],
        ),
    ])
