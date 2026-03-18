#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <string>
#include <stdexcept>
#include <vector>

#include "controller/controller.hpp"

namespace controller
{

    class JoystickController : public Controller
    {

    public:
        JoystickController()
            : Controller("joystick_controller_node")
        {
            param_callback_handle_ = this->add_on_set_parameters_callback(
                std::bind(&JoystickController::on_parameter_change, this, std::placeholders::_1));

            this->declare_parameter("axis", 4);

            joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
                "/joy", 10,
                std::bind(&JoystickController::joy_callback, this, std::placeholders::_1));
        }

    private:
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
        int joy_axis_;
        rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;

        void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
        {
            float motion = 0.0f;

            // Check if axis indices are within range
            if (joy_axis_ >= static_cast<int>(msg->axes.size()))
            {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                      "Axis index %d is out of bounds! Joy message has %zu axes.",
                                      joy_axis_, msg->axes.size());
                return;
            }

            motion = msg->axes.at(joy_axis_);

            // Wrap control call in try-catch to catch any runtime exceptions
            this->control(motion);
        }

        rcl_interfaces::msg::SetParametersResult on_parameter_change(
            const std::vector<rclcpp::Parameter> &parameters)
        {
            for (const auto &param : parameters)
            {
                if (param.get_name() == "axis")
                {
                    joy_axis_ = param.as_int();
                    RCLCPP_INFO(this->get_logger(), "Updated axis: %d", joy_axis_);
                }
            }

            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            result.reason = "Joystick axis parameters updated";
            return result;
        }
    };

}

// Main function
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<controller::JoystickController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}