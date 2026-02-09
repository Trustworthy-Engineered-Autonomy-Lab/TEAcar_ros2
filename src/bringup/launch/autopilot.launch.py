# bringup/launch/autopilot.launch.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bringup_share = get_package_share_directory('bringup')
    drive_launch = os.path.join(bringup_share, 'launch', 'drive.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(drive_launch)
        ),

        SetEnvironmentVariable(
            name="GSCAM_CONFIG",
            value=(
                "nvarguscamerasrc "
                "! video/x-raw(memory:NVMM), width=3280, height=2464, format=NV12, framerate=21/1 "
                "! nvvidconv flip-method=2 "
                "! video/x-raw, width=224, height=224, format=BGRx "
                "! videocrop left=0 right=0 top=80 bottom=0 "
                "! videoconvert"
            ),
        ),

        Node(
            package="gscam",
            executable="gscam_node",
            name="camera",
            output="screen",
        ),

        Node(
            package="controller",
            executable="nn_controller",
            name="nn_controller_node",
            output="screen",
            parameters=[
                {"steer_ratio": -1.0},
                {"backend": "tensorrt"},
                {"model_file": os.path.join(
                    get_package_share_directory('bringup'),
                    "models",
                    "best.onnx",
                )},
                {"input_name": "image"},
                {"output_name": "steer"},
            ],
            remappings=[
                ("image_raw", "/camera/image_raw"),
            ],
        ),
    ])