#include "rclcpp/rclcpp.hpp"
#include "nn_controller/nn_controller_node.hpp"

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<nnc::NNControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
