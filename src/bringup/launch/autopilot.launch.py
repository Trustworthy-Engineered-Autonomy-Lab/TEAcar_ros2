from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Base driving launch
    drive_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bringup'),
                'launch',
                'drive.launch.py',
            )
        )
    )

    # Parameters for nn_controller (from nn_controller/config/nn_controller.yaml)
    nn_controller_config = os.path.join(
        get_package_share_directory('nn_controller'),
        'config',
        'nn_controller.yaml',
    )

    nn_controller_with_params = Node(
        package='controller',
        executable='nn_controller_node',
        name='nn_controller',
        output='screen',
        parameters=[nn_controller_config],
    )

    return LaunchDescription([
        drive_launch,
        nn_controller_with_params,
    ])
