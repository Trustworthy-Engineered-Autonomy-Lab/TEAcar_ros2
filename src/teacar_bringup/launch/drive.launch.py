from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    teacar_bringup_pkg = FindPackageShare('teacar_bringup')

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([teacar_bringup_pkg, 'config', 'drive_config.yaml'])
    )

    camera_launch = IncludeLaunchDescription(
        PathJoinSubstitution([teacar_bringup_pkg, 'launch', 'camera.launch.py']),
        launch_arguments={
            "config_file": LaunchConfiguration("config_file")
        }.items()
    )

    # Joystick driver node
    joy_node = Node(
        package="joy_linux",
        executable="joy_linux_node",
        name="joy_linux_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
        ]
    )
    
    # Steer controllers
    steer_controller_nodes = [
        Node(
            package="controller",
            executable="joystick_controller",
            name="joystick_steer_controller_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file")
            ],
            remappings=[
                ("/motion", "/steer")
            ]
        ),
        Node(
            package="controller",
            executable="param_controller",
            name="param_steer_controller_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file")
            ],
            remappings=[
                ("/motion", "/steer")
            ]
        )
    ]

    # Throttle controllers
    throttle_controller_nodes = [
        Node(
            package="controller",
            executable="joystick_controller",
            name="joystick_throttle_controller_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file")
            ],
            remappings=[
                ("/motion", "/throttle")
            ]
        ),
        Node(
            package="controller",
            executable="param_controller",
            name="param_throttle_controller_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file")
            ],
            remappings=[
                ("/motion", "/throttle")
            ]
        )
    ]

    # Steer actuator
    steer_actuator_node = Node(
        package="actuator",
        executable="pwm_based_actuator",
        name="steer_actuator_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file")
        ],
        remappings=[
            ("/motion", "/steer"),
            ("/combined_motion", "/combined_steer")
        ]
    )

    # Throttle actuator
    throttle_actuator_node = Node(
        package="actuator",
        executable="pwm_based_actuator",
        name="throttle_actuator_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file")
        ],
        remappings=[
            ("/motion", "/throttle"),
            ("/combined_motion", "/combined_throttle")
        ]
    )

    # PCA9685 driver
    pca9685_driver_node = Node(
        package="actuator",
        executable="pca9685_driver",
        name="pca9685_driver_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        camera_launch,
        joy_node,
        *steer_controller_nodes,
        *throttle_controller_nodes,
        steer_actuator_node,
        throttle_actuator_node,
        pca9685_driver_node
    ])