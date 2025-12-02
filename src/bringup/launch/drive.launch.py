from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Basic driving stack: camera only
    return LaunchDescription([
        Node(
            package='image_tools',
            executable='cam2image',
            name='camera',
            output='screen'
        ),
    ])
