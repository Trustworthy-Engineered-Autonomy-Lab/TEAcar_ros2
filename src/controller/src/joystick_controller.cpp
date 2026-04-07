#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <string>
#include <vector>
#include <algorithm>
#include <functional>

template<typename T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi)
{
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

#include "controller/controller.hpp"

class JoystickController : public controller::Controller
{
public:
  JoystickController()
  : controller::Controller("joystick_controller_node"),
    steer_axis_(0),
    decel_button_(0),
    stop_button_(1),
    init_speed_button_(2),
    accel_button_(3),
    initial_speed_(0.20),
    speed_step_(0.02),
    min_speed_(0.0),
    max_speed_(0.40),
    current_speed_(0.0)
  {
    // 方向控制轴
    this->declare_parameter<int>("steer_axis", 0);

    // 按键映射
    this->declare_parameter<int>("decel_button", 0);
    this->declare_parameter<int>("stop_button", 1);
    this->declare_parameter<int>("init_speed_button", 2);
    this->declare_parameter<int>("accel_button", 3);

    // 巡航速度参数
    this->declare_parameter<double>("initial_speed", 0.20);
    this->declare_parameter<double>("speed_step", 0.02);
    this->declare_parameter<double>("min_speed", 0.0);
    this->declare_parameter<double>("max_speed", 0.40);

    // 读取参数
    this->get_parameter("steer_axis", steer_axis_);

    this->get_parameter("decel_button", decel_button_);
    this->get_parameter("stop_button", stop_button_);
    this->get_parameter("init_speed_button", init_speed_button_);
    this->get_parameter("accel_button", accel_button_);

    this->get_parameter("initial_speed", initial_speed_);
    this->get_parameter("speed_step", speed_step_);
    this->get_parameter("min_speed", min_speed_);
    this->get_parameter("max_speed", max_speed_);

    // 初始当前速度设为 0，等按下“初速度键”再启动
    current_speed_ = 0.0;

    param_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&JoystickController::on_parameter_change, this, std::placeholders::_1));

    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10,
      std::bind(&JoystickController::joy_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "==============================================");
    RCLCPP_INFO(this->get_logger(), "Joystick cruise controller started");
    RCLCPP_INFO(this->get_logger(), "steer_axis        = %d", steer_axis_);
    RCLCPP_INFO(this->get_logger(), "decel_button      = %d", decel_button_);
    RCLCPP_INFO(this->get_logger(), "stop_button       = %d", stop_button_);
    RCLCPP_INFO(this->get_logger(), "init_speed_button = %d", init_speed_button_);
    RCLCPP_INFO(this->get_logger(), "accel_button      = %d", accel_button_);
    RCLCPP_INFO(this->get_logger(), "initial_speed     = %.3f", initial_speed_);
    RCLCPP_INFO(this->get_logger(), "speed_step        = %.3f", speed_step_);
    RCLCPP_INFO(this->get_logger(), "min_speed         = %.3f", min_speed_);
    RCLCPP_INFO(this->get_logger(), "max_speed         = %.3f", max_speed_);
    RCLCPP_INFO(this->get_logger(), "current_speed     = %.3f", current_speed_);
    RCLCPP_INFO(this->get_logger(), "==============================================");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  int steer_axis_;

  int decel_button_;
  int stop_button_;
  int init_speed_button_;
  int accel_button_;

  double initial_speed_;
  double speed_step_;
  double min_speed_;
  double max_speed_;
  double current_speed_;

  std::vector<int> last_buttons_;

  bool is_button_pressed(const sensor_msgs::msg::Joy::SharedPtr msg, int button_index)
  {
    if (button_index < 0) {
      return false;
    }

    if (static_cast<size_t>(button_index) >= msg->buttons.size()) {
      return false;
    }

    if (last_buttons_.size() != msg->buttons.size()) {
      last_buttons_.assign(msg->buttons.size(), 0);
    }

    return (msg->buttons[button_index] == 1 && last_buttons_[button_index] == 0);
  }

  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    if (last_buttons_.size() != msg->buttons.size()) {
      last_buttons_.assign(msg->buttons.size(), 0);
    }

    // 1. 按键控制速度
    if (is_button_pressed(msg, decel_button_)) {
      current_speed_ -= speed_step_;
      current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
      RCLCPP_INFO(this->get_logger(), "[Button %d] decelerate -> current_speed = %.3f",
                  decel_button_, current_speed_);
    }

    if (is_button_pressed(msg, stop_button_)) {
      current_speed_ = 0.0;
      RCLCPP_INFO(this->get_logger(), "[Button %d] stop -> current_speed = %.3f",
                  stop_button_, current_speed_);
    }

    if (is_button_pressed(msg, init_speed_button_)) {
      current_speed_ = initial_speed_;
      current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
      RCLCPP_INFO(this->get_logger(), "[Button %d] set initial speed -> current_speed = %.3f",
                  init_speed_button_, current_speed_);
    }

    if (is_button_pressed(msg, accel_button_)) {
      current_speed_ += speed_step_;
      current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
      RCLCPP_INFO(this->get_logger(), "[Button %d] accelerate -> current_speed = %.3f",
                  accel_button_, current_speed_);
    }

    // 2. 摇杆控制转向
    float steer = 0.0f;

    if (steer_axis_ >= 0 && static_cast<size_t>(steer_axis_) < msg->axes.size()) {
      steer = static_cast<float>(msg->axes.at(steer_axis_));
    } else {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Steer axis index %d is out of bounds! Joy message has %zu axes.",
        steer_axis_, msg->axes.size());
    }

    // 3. 发布控制命令
    try {
      this->control(static_cast<float>(current_speed_), steer);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Exception in control(): %s", e.what());
    }

    // 4. 保存本次按钮状态
    last_buttons_ = msg->buttons;
  }

  rcl_interfaces::msg::SetParametersResult on_parameter_change(
    const std::vector<rclcpp::Parameter> &parameters)
  {
    for (const auto &param : parameters) {
      if (param.get_name() == "steer_axis") {
        steer_axis_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Updated steer_axis to %d", steer_axis_);
      }
      else if (param.get_name() == "decel_button") {
        decel_button_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Updated decel_button to %d", decel_button_);
      }
      else if (param.get_name() == "stop_button") {
        stop_button_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Updated stop_button to %d", stop_button_);
      }
      else if (param.get_name() == "init_speed_button") {
        init_speed_button_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Updated init_speed_button to %d", init_speed_button_);
      }
      else if (param.get_name() == "accel_button") {
        accel_button_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Updated accel_button to %d", accel_button_);
      }
      else if (param.get_name() == "initial_speed") {
        initial_speed_ = param.as_double();
        current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
        RCLCPP_INFO(this->get_logger(), "Updated initial_speed to %.3f", initial_speed_);
      }
      else if (param.get_name() == "speed_step") {
        speed_step_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Updated speed_step to %.3f", speed_step_);
      }
      else if (param.get_name() == "min_speed") {
        min_speed_ = param.as_double();
        current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
        RCLCPP_INFO(this->get_logger(), "Updated min_speed to %.3f", min_speed_);
      }
      else if (param.get_name() == "max_speed") {
        max_speed_ = param.as_double();
        current_speed_ = clamp(current_speed_, min_speed_, max_speed_);
        RCLCPP_INFO(this->get_logger(), "Updated max_speed to %.3f", max_speed_);
      }
    }

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "Parameters updated successfully";
    return result;
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<JoystickController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}