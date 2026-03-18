#include "controller/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

namespace controller
{

    class ParamControllerNode : public Controller
    {
    public:
        ParamControllerNode() : Controller("param_controller_node")
        {
            // Set up parameter change callback
            param_callback_handle_ = this->add_on_set_parameters_callback(
                std::bind(&ParamControllerNode::on_parameter_change, this, std::placeholders::_1));

            this->declare_parameter("motion", 0.0);
        }

    private:
        float motion_;

        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
        rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &parameters)
        {
            for (const auto &param : parameters)
            {
                if (param.get_name() == "motion")
                {
                    motion_ = param.as_double();
                    this->control(motion_);
                    RCLCPP_INFO(this->get_logger(), "Updated motion: %.2f", motion_);
                }
            }

            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            result.reason = "Updated parameters successfully";
            return result;
        }
    };

}

// Entry point
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<controller::ParamControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
