#include "actuator.hpp" // Base class

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

namespace actuator
{

    class PWMBasedActuatorNode : public Actuator
    {
    public:
        PWMBasedActuatorNode() : Actuator("pwm_based_actuator")
        {
            // Parameter update callback
            param_callback_handle_ = this->add_on_set_parameters_callback(
                std::bind(&PWMBasedActuatorNode::on_parameter_change, this, std::placeholders::_1));

            this->declare_parameter("min_pulsewidth", 1000);
            this->declare_parameter("mid_pulsewidth", 1500);
            this->declare_parameter("max_pulsewidth", 2000);
            this->declare_parameter("pwm_channel", 0);
        }

    protected:
        virtual void actuate(float combined_motion) override
        {
            float pulse_width = 0.0f;
            if (combined_motion >= 0)
                pulse_width = combined_motion * (max_pw_ - mid_pw_) + mid_pw_;
            else
                pulse_width = combined_motion * (mid_pw_ - min_pw_) + mid_pw_;

            if (pulse_width_pub_)
            {
                auto msg = std_msgs::msg::UInt32();
                msg.data = static_cast<uint32_t>(pulse_width);
                pulse_width_pub_->publish(msg);
                // RCLCPP_DEBUG(this->get_logger(), "Sent pulse width %d to channel %d", pulse_width, pwm_channel_);
            }
        }

    private:
        int min_pw_ = 1000;
        int max_pw_ = 2000;
        int mid_pw_ = 1500;
        int pwm_channel_ = 0;

        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
        rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pulse_width_pub_;

        rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &parameters)
        {
            for (const auto &param : parameters)
            {
                const std::string &param_name = param.get_name();
                if (param_name == "min_pulsewidth")
                {
                    min_pw_ = check_pulsewidth(min_pw_, param.as_int());
                }
                else if (param_name == "mid_pulsewidth")
                {
                    mid_pw_ = check_pulsewidth(mid_pw_, param.as_int());
                }
                else if (param_name == "max_pulsewidth")
                {
                    max_pw_ = check_pulsewidth(max_pw_, param.as_int());
                }
                else if (param_name == "pwm_channel")
                {
                    set_channel(param.as_int());
                }
            }

            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            result.reason = "Updated parameters successfully";
            return result;
        }

        void set_channel(int new_pwm_channel)
        {
            if (new_pwm_channel < 0 || new_pwm_channel > 15)
            {
                RCLCPP_ERROR(this->get_logger(), "Invalid PWM channel %d", new_pwm_channel);
                return;
            }

            if (pulse_width_pub_ && (new_pwm_channel != pwm_channel_))
            {
                pulse_width_pub_.reset();
            }
            pwm_channel_ = new_pwm_channel;

            RCLCPP_INFO(this->get_logger(), "Updated PWM channel: %d", pwm_channel_);
            if (!pulse_width_pub_)
            {
                pulse_width_pub_ = this->create_publisher<std_msgs::msg::UInt32>(
                    "/pca9685/channel" + std::to_string(pwm_channel_) + "/pulse_width", 10);
            }
        }

        int check_pulsewidth(int old_value, int new_value)
        {
            if (new_value < 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Invalid pulse width %d", new_value);
                return old_value;
            }

            if (!(min_pw_ <= mid_pw_ && mid_pw_ <= max_pw_))
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "Invalid pulse width ordering: expected min <= mid <= max, got min=%d, mid=%d, max=%d",
                    min_pw_, mid_pw_, max_pw_);
            }
            else
            {
                RCLCPP_INFO(this->get_logger(),
                            "Updated pulse width: min=%d, mid=%d, max=%d",
                            min_pw_, mid_pw_, max_pw_);
            }

            return new_value;
        }
    };

}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<actuator::PWMBasedActuatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
