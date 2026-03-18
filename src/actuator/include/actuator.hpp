// actuator_node.hpp
#ifndef ACTUATOR__ACTUATOR_NODE_HPP_
#define ACTUATOR__ACTUATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include "teacar_msgs/msg/motioncmd.hpp"

#include <unordered_map>
#include <array>
#include <string>
#include <algorithm>

namespace actuator
{
    class Actuator : public rclcpp::Node
    {

    public:
        Actuator(const std::string &node_name) : rclcpp::Node(node_name)
        {
            // Parameter update callback
            param_callback_handle_ = this->add_on_set_parameters_callback(
                std::bind(&Actuator::on_parameter_change, this, std::placeholders::_1));

            // Declare and get parameter
            this->declare_parameter("control_frequency", 50);

            // Create publisher and subscriber
            combined_motion_pub_ = this->create_publisher<teacar_msgs::msg::Motioncmd>("/combined_motion", 10);

            motion_sub_ = this->create_subscription<teacar_msgs::msg::Motioncmd>(
                "/motion", 10,
                std::bind(&Actuator::motion_callback, this, std::placeholders::_1));
        }

    protected:
        virtual void actuate(float combined_motion) {}

    private:
        int control_frequency_ = 50;
        rclcpp::Publisher<teacar_msgs::msg::Motioncmd>::SharedPtr combined_motion_pub_;
        rclcpp::Subscription<teacar_msgs::msg::Motioncmd>::SharedPtr motion_sub_;
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
        std::unordered_map<std::string, float> motion_map_;

        void timer_callback()
        {
            float combined_motion = 0.0f;

            for (const auto &pair : motion_map_)
            {
                combined_motion += pair.second;
            }

            // Use manual clamping instead of std::clamp for compatibility
            combined_motion = std::clamp(combined_motion, -1.0f, 1.0f);

            teacar_msgs::msg::Motioncmd msg;
            msg.header.stamp = this->get_clock()->now();
            msg.source = this->get_name();
            msg.value = combined_motion;

            if (combined_motion_pub_)
            {
                combined_motion_pub_->publish(msg);
                RCLCPP_DEBUG(this->get_logger(), "Published combined motion: %.2f", combined_motion);
            }

            actuate(combined_motion);
        }

        void motion_callback(const teacar_msgs::msg::Motioncmd::SharedPtr msg)
        {
            RCLCPP_DEBUG(this->get_logger(), "Received motion: %.2f from node: %s",
                         msg->value, msg->source);

            motion_map_[msg->source] = msg->value;
        }

        rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &parameters)
        {
            for (const auto &param : parameters)
            {
                if (param.get_name() == "control_frequency")
                {
                    set_control_frequency(param.as_int());
                }
            }
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            result.reason = "Updated parameters successfully";
            return result;
        }

        void set_control_frequency(int new_freq)
        {
            // Check if the control frequency is valid
            if (new_freq <= 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Invalid control frequency %d Hz. Ignored", new_freq);
                return;
            }

            // If the frequency is changed, stop and remove the old timer
            if ((new_freq != control_frequency_) && timer_)
            {
                timer_->reset();
                timer_->cancel();
                timer_.reset();
            }

            control_frequency_ = new_freq;

            // Create the timer
            if (!timer_)
            {
                timer_ = this->create_wall_timer(
                    std::chrono::duration<double>(1.0 / control_frequency_),
                    std::bind(&Actuator::timer_callback, this));
            }

            RCLCPP_INFO(this->get_logger(), "Updated control frequency: %d Hz", control_frequency_);
        }
    };
}

#endif // ACTUATOR__ACTUATOR_NODE_HPP_