#include <rclcpp/rclcpp.hpp>
#include "nn_controller/nn_controller_node.hpp"
int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<nnc::NNControllerNode>());
  rclcpp::shutdown();
  return 0;
}
