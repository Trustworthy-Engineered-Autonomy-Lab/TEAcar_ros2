#ifndef CONTROLLER__CONTROLLER_HPP_
#define CONTROLLER__CONTROLLER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <teacar_msgs/msg/motioncmd.hpp>

#include <cmath>
#include <string>

namespace controller
{
    class Controller : public rclcpp::Node
    {
    public:
        Controller(const std::string &node_name) : rclcpp::Node(node_name)
        {
            publisher_ = this->create_publisher<teacar_msgs::msg::Motioncmd>("/motion", 10);
            callback_handle_ = this->add_on_set_parameters_callback(std::bind(&Controller::parameterCallback,
                                                                              this, std::placeholders::_1));

            this->declare_parameter("positive_ratio", 1.0);
            this->declare_parameter("negative_ratio", 1.0);
        }

        void control(float motion)
        {
            auto msg = teacar_msgs::msg::Motioncmd();
            msg.header.stamp = this->get_clock()->now();
            msg.source = this->get_name();

            if (motion > 0)
            {
                msg.value = motion * pos_ratio_;
            }
            else
            {
                msg.value = motion * neg_ratio_;
            }

            publisher_->publish(msg);

            RCLCPP_DEBUG(this->get_logger(), "Sent motion: %.2f from node %s",
                         msg.value, this->get_name());
        }

    private:
        float pos_ratio_ = 1.0f;
        float neg_ratio_ = 1.0f;

        rclcpp::Publisher<teacar_msgs::msg::Motioncmd>::SharedPtr publisher_;
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_handle_;

        rcl_interfaces::msg::SetParametersResult parameterCallback(const std::vector<rclcpp::Parameter> &parameters)
        {
            for (const auto &param : parameters)
            {
                if (param.get_name() == "positive_ratio")
                {
                    pos_ratio_ = param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated positive ratio: %f", pos_ratio_);
                }
                else if (param.get_name() == "negative_ratio")
                {
                    neg_ratio_ = param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated negative ratio: %f", neg_ratio_);
                }
            }
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            result.reason = "Parameters updated successfully";
            return result;
        }
    };

}

#endif // CONTROLLER__CONTROLLER_HPP_