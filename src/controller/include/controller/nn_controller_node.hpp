#pragma once
#include <optional>
#include <string>
#include <memory>
#include <cstring>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "donkeycar_msgs/msg/motion_cmd.hpp"
#include "nn_controller/inferencer_api.hpp"

namespace nnc {
class NNControllerNode : public rclcpp::Node {
public:
  explicit NNControllerNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions());
private:
  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<donkeycar_msgs::msg::MotionCmd>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Clock clock_{RCL_ROS_TIME};

  // Params
  double control_rate_hz_ = 50.0;
  double throttle_ratio_  = 1.0;
  double steer_ratio_     = 1.0;
  std::string image_topic_ = "/camera/image_raw";
  std::string backend_     = "tensorflow";
  std::string model_file_  = "/tmp/dummy.onnx";
  std::string input_name_  = "input";
  std::string output_name_ = "outputs";
  int roi_x_ = 0, roi_y_ = 0, roi_w_ = 224, roi_h_ = 144;

  // State
  sensor_msgs::msg::Image::ConstSharedPtr last_image_;
  cv::Rect roi_;

  // Inference
  InferencerPtr inferencer_;
  bool   backend_ready_ = false;
  void*  input_buffer_  = nullptr;
  void*  output_buffer_ = nullptr;
  size_t input_buffer_size_  = 0;
  size_t output_buffer_size_ = 0;

  // Methods
  void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void tick();
  void resizeTimer();
  bool ensureBackendReadyOnce();
  donkeycar_msgs::msg::MotionCmd makeCmd(float throttle, float steer);
};
} // namespace nnc
