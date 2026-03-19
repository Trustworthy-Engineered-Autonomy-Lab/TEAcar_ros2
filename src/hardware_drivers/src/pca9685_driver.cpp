#include "hardware_drivers/pca9685.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <memory>

class PCA9685DriverNode : public rclcpp::Node
{
public:
    PCA9685DriverNode() : rclcpp::Node("pca9685_driver_node")
    {
        pca_ = std::make_unique<PCA9685>();
        // Parameter update callback
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&PCA9685DriverNode::on_parameter_change, this, std::placeholders::_1));

        this->declare_parameter("bus_device", std::string("/dev/i2c-1"));
        this->declare_parameter("pwm_frequency", 60);

        for (int i = 0; i < 16; i++)
        {
            auto pulse_width_sub_ = this->create_subscription<std_msgs::msg::UInt32>(
                "/pca9685/channel" + std::to_string(i) + "/pulse_width", 10,
                [this, channel = i](const std_msgs::msg::UInt32::SharedPtr msg)
                {
                    pulse_width_callback(channel, msg);
                });
            pulse_width_subs_.push_back(pulse_width_sub_);
        }

        pca9685_monitor_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&PCA9685DriverNode::pca9685_monitor_callback, this));
    }

    ~PCA9685DriverNode()
    {
        if (pca_)
        {
            pca_->close();
        }
    }

private:
    std::unique_ptr<PCA9685> pca_;
    std::string bus_device_;
    int pwm_freq_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rclcpp::TimerBase::SharedPtr pca9685_monitor_;
    std::vector<rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr> pulse_width_subs_;

    rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &parameters)
    {
        for (const auto &param : parameters)
        {
            const std::string &param_name = param.get_name();
            if (param_name == "bus_device")
            {
                std::string new_bus_device = param.as_string();
                if (new_bus_device != bus_device_)
                {
                    if (!pca_->is_opened())
                    {
                        // If pca9685 is not opened, update the parameter than wait for the monitor to open it.
                        bus_device_ = new_bus_device;
                        RCLCPP_INFO(this->get_logger(), "Bus device is set to %s", bus_device_.c_str());
                    }
                    else
                    {
                        // Otherwise create a new instance and try opening it.
                        std::unique_ptr<PCA9685> new_pca = std::make_unique<PCA9685>();
                        if (!new_pca->open(new_bus_device, pwm_freq_))
                        {
                            RCLCPP_ERROR(this->get_logger(), "Could not open pca9685 on %s. Ignored", new_bus_device.c_str());
                        }
                        else
                        {
                            RCLCPP_INFO(this->get_logger(), "Updated bus device: %s. Opened pca9685 on %s.", new_bus_device.c_str(), new_bus_device.c_str());
                            bus_device_ = new_bus_device;
                            pca_ = std::move(new_pca);
                        }
                    }
                }
            }
            else if (param_name == "pwm_frequency")
            {
                pwm_freq_ = param.as_int();
                if (!pca_->is_opened())
                {
                    RCLCPP_ERROR(this->get_logger(), "Updated PWM frequency: %d. But failed to apply setting: pca9685 is not opened yet.", pwm_freq_);
                }
                else if (!pca_->set_pwm_freq(pwm_freq_))
                {
                    RCLCPP_ERROR(this->get_logger(), "Updated PWM frequency: %d. But failed to apply setting: %s", pwm_freq_, pca_->get_error().c_str());
                }
                else
                {
                    RCLCPP_INFO(this->get_logger(), "Updated PWM frequencty: %d", pwm_freq_);
                }
            }
        }

        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "Updated parameters successfully";
        return result;
    }

    void pca9685_monitor_callback()
    {
        if (!pca_->is_opened())
        {
            if (pca_->open(bus_device_, pwm_freq_))
                RCLCPP_INFO(this->get_logger(), "Opened pca9685 on %s", bus_device_.c_str());
            else
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to open pca9685 on %s: %s. Will retry", bus_device_.c_str(), pca_->get_error().c_str());
        }
    }

    void pulse_width_callback(int channel, const std_msgs::msg::UInt32::SharedPtr msg)
    {
        if (pca_->is_opened())
        {
            float period = 1000000.0f / pwm_freq_;
            float duty_cycle = msg->data / period;
            if (!pca_->set_pwm_dutycycle(channel, duty_cycle))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to set duty cycle %.2f for channel %d: %s", duty_cycle, channel, pca_->get_error().c_str());
                pca_->close();
            }
            else
            {
                RCLCPP_DEBUG(this->get_logger(), "Set duty cycle %.2f for channel %d", duty_cycle, channel);
            }
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCA9685DriverNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}