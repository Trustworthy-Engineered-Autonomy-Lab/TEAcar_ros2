from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('nn_controller'),'config','nn_controller.yaml')
    return LaunchDescription([
        Node(package='nn_controller', executable='nn_controller_node',
             name='nn_controller_node', output='screen', parameters=[cfg])
    ])
