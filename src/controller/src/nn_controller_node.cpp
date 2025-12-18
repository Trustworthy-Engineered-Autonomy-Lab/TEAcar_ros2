#include "interfaces_msg/msg/motion_cmd.hpp"
#include "controller/nn_controller_node.hpp"
#include <algorithm>

using std::placeholders::_1;

namespace nnc {

NNControllerNode::NNControllerNode(const rclcpp::NodeOptions& opts)
: rclcpp::Node("nn_controller_node", opts) {
  control_rate_hz_ = declare_parameter<double>("control_rate_hz", control_rate_hz_);
  throttle_ratio_  = declare_parameter<double>("throttle_ratio",  throttle_ratio_);
  steer_ratio_     = declare_parameter<double>("steer_ratio",     steer_ratio_);
  image_topic_     = declare_parameter<std::string>("image_topic", image_topic_);
  backend_         = declare_parameter<std::string>("backend",     backend_);
  model_file_      = declare_parameter<std::string>("model_file",  model_file_);
  input_name_      = declare_parameter<std::string>("input_name",  input_name_);
  output_name_     = declare_parameter<std::string>("output_name", output_name_);
  roi_x_           = declare_parameter<int>("roi.x",     roi_x_);
  roi_y_           = declare_parameter<int>("roi.y",     roi_y_);
  roi_w_           = declare_parameter<int>("roi.width", roi_w_);
  roi_h_           = declare_parameter<int>("roi.height",roi_h_);

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(), std::bind(&NNControllerNode::imageCb, this, _1));
  cmd_pub_ = create_publisher<interfaces_msg::msg::MotionCmd>("/motion_cmd", 10);
  resizeTimer();

  RCLCPP_INFO(get_logger(), "nn_controller_node up. backend=%s, image_topic=%s",
              backend_.c_str(), image_topic_.c_str());
}

void NNControllerNode::resizeTimer() {
  if (control_rate_hz_ <= 0.0) control_rate_hz_ = 10.0;
  auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::milliseconds>(period),
    std::bind(&NNControllerNode::tick, this));
}

void NNControllerNode::imageCb(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  last_image_ = msg;
  const int W = static_cast<int>(msg->width), H = static_cast<int>(msg->height);
  const int w = std::clamp(roi_w_, 1, W), h = std::clamp(roi_h_, 1, H);
  roi_ = cv::Rect(roi_x_, roi_y_, w, h) & cv::Rect(0,0,W,H);
}

bool NNControllerNode::ensureBackendReadyOnce() {
  if (backend_ready_) return true;

  inferencer_ = std::make_shared<nnc::DynamicInferencer>(backend_);

  if (!inferencer_->loadModel(model_file_)) {
    RCLCPP_ERROR(get_logger(), "loadModel failed: %s", inferencer_->getErrorString().c_str());
    return false;
  }
  output_buffer_size_ = inferencer_->getOutputBuffer(output_name_, &output_buffer_);
  if (output_buffer_size_ != sizeof(float)) {
    RCLCPP_FATAL(get_logger(), "output tensor must be 4 bytes (got %zu)", output_buffer_size_);
    rclcpp::shutdown();
    return false;
  }
  input_buffer_size_ = inferencer_->getInputBuffer(input_name_, &input_buffer_);
  if (input_buffer_size_ == 0) {
    RCLCPP_ERROR(get_logger(), "Input buffer size is 0");
    return false;
  }

  RCLCPP_INFO(get_logger(), "Backend ready. input=%zu bytes, output=%zu bytes",
              input_buffer_size_, output_buffer_size_);
  backend_ready_ = true;
  return true;
}

interfaces_msg::msg::MotionCmd NNControllerNode::makeCmd(float throttle, float steer) {
  interfaces_msg::msg::MotionCmd cmd;
  cmd.header.stamp = clock_.now();
  cmd.header.frame_id = get_fully_qualified_name();
  cmd.throttle = static_cast<float>(throttle_ratio_ * throttle);
  cmd.steer    = static_cast<float>(steer_ratio_ * steer);
  return cmd;
}

void NNControllerNode::tick() {
  if (!ensureBackendReadyOnce()) return;
  if (!last_image_) return;

  // Accept mono8, rgb8, and bgr8 (cam2image default)
  int channels = sensor_msgs::image_encodings::numChannels(last_image_->encoding);
  if (channels != 1 && channels != 3) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "Unsupported encoding: %s", last_image_->encoding.c_str());
    return;
  }

  const size_t roi_bytes = static_cast<size_t>(roi_.area()) * 4 * channels;
  if (roi_bytes != input_buffer_size_) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
      "ROI bytes %zu != model input bytes %zu (use 224x144 RGB/BGR for mock)",
      roi_bytes, input_buffer_size_);
    return;
  }

  cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(last_image_, last_image_->encoding);
  cv::Mat roi_u8 = img_ptr->image(roi_);
  cv::Mat roi_rgb;

  // If BGR8, convert to RGB so future real models are happy.
  if (last_image_->encoding == sensor_msgs::image_encodings::BGR8) {
    cv::cvtColor(roi_u8, roi_rgb, cv::COLOR_BGR2RGB);
  } else {
    roi_rgb = roi_u8; // mono8 or rgb8 already OK
  }

  cv::Mat input_float;
  roi_rgb.convertTo(input_float, channels==1 ? CV_32FC1 : CV_32FC3);
  std::memcpy(input_buffer_, input_float.data, input_buffer_size_);

  if (!inferencer_->infer()) {
    RCLCPP_ERROR(get_logger(), "infer() failed: %s", inferencer_->getErrorString().c_str());
    backend_ready_ = false;
    return;
  }

  float steer_est = *reinterpret_cast<float*>(output_buffer_);
  cmd_pub_->publish(makeCmd(0.0f, steer_est));
}

} // namespace nnc

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<nnc::NNControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
