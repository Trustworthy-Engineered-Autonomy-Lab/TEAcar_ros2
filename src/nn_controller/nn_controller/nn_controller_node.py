#!/usr/bin/env python3
import importlib
from typing import Optional
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
# from interfaces_msg.msg import MotionCmd

class NNController(Node):
    def __init__(self):
        super().__init__('nn_controller')

        # Parameters
        self.declare_parameter('plugin_module', 'nn_controller.plugins.dummy_inferencer')
        self.declare_parameter('plugin_class', 'DummyInferencer')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/steer_angle')

        plugin_module = self.get_parameter('plugin_module').get_parameter_value().string_value
        plugin_class  = self.get_parameter('plugin_class').get_parameter_value().string_value
        image_topic   = self.get_parameter('image_topic').get_parameter_value().string_value
        output_topic  = self.get_parameter('output_topic').get_parameter_value().string_value

        # Load plugin
        mod = importlib.import_module(plugin_module)
        InferencerClass = getattr(mod, plugin_class)
        self.inferencer = InferencerClass(self.get_logger())
        self.get_logger().info(f'Loaded inferencer: {plugin_module}.{plugin_class}')

        # Setup publishers/subscribers
        self.bridge = CvBridge()
        self.pub_steer = self.create_publisher(Float32, output_topic, 10)
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.last_ok: Optional[float] = None

    def on_image(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            steer_angle = self.inferencer.infer(cv_image)
            self.last_ok = steer_angle
            out = Float32()
            out.data = float(steer_angle)
            self.pub_steer.publish(out)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main():
    rclpy.init()
    node = NNController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

