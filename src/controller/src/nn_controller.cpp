// ROS2 nn_controller_node.cpp
#include "controller/controller.hpp"
#include "rclcpp/rclcpp.hpp"

#include <stdexcept>

#include "inferencer/inferencer.h"

class NNControllerNode() : controller::Controller {
public:
    NNControllerNode() : controller::Controller("nn_controller_node"), status(Status::INIT_BACKEND), inferencer(nullptr) {
        // Define default parameters
        this->declare_parameter<std::string>("backend", "tensorflow");
        this->declare_parameter<std::string>("model_file", "");
        this->declare_parameter<std::string>("output_name", "outputs");
        this->declare_parameter<std::string>("input_name", "inputs");
        this->declare_parameter<int>("roi/x", 0);
        this->declare_parameter<int>("roi/y", 0);

        // Load parameters once at startup
        this->get_parameter("backend", backend_);
        this->get_parameter("model_file", model_file_);
        this->get_parameter("output_name", output_name_);
        this->get_parameter("input_name", input_name_);

        constexpr interval = std::chrono::seconds(1);
        const auto init_callback = [this](){this->init()};
        retry_init_timer = this->create_timer(interval, timer_callback);

        RCLCPP_INFO(this->get_logger(), "NNControllerNode initialized.");
    }

private:
    // Parameters
    std::string backend_;
    std::string model_file_;
    std::string output_name_;
    std::string input_name_;

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr image_sub;
    rclcpp::TimerBase::SharedPtr retry_init_timer;

    enum class Status {
        INIT_BACKEND,
        LOAD_MODEL,
        ALLOC_OUTPUT,
        ALLOC_INPUT,
        WAIT_IMAGE,
        RUNNING,
    };

    Status status;
    std::shared_ptr<inferencer::Inferencer> inferencer;

    void initCallback() {
        switch (this->status) {
        case Status::INIT_BACKEND: {
            try {
                // Dynamically load the inferencer backend
                this->inferencer = std::make_shared<inferencer::Inferencer>(this->backend_);
            }
            catch(const std::runtime_error& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize the %s backend: %s. Will retry", this->backend.c_str(), e.what());
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully initialize %s backend", this->backend.c_str());
            this->status = Status::LOAD_MODEL;
        }
        case Status::LOAD_MODEL: {
            if(this->model_file_.empty()) {
                // TODO: default model file path logic
                RCLCPP_ERROR(this->get_logger(), "Model file parameter is empty. Will retry");
                return;
            }

            bool result;
            try {
                // TODO: make sure arguments are correct
                result = this->inferencer->load_model(model_file_);
            }
            catch(const std::runtime_error& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load model file %s: %s. Will retry", this->model_file_.c_str(), e.what());
                return;
            }

            if(!result) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load model file %s: %s. Will retry", this->model_file_.c_str(), this->inferencer->get_error_string());
                return;
            }
            
            RCLPP_INFO(this->get_logger(), "Successfully load model file %s", this->model_file_.c_str());
            status = Status::ALLOC_OUTPUT;
        }
        case Status::ALLOC_OUTPUT: {
            try {
                outputBufferSize = this->inferencer->get_output_buffer(output_name_, &outputBuffer);
            }
            catch(const std::runtime_error& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate ouput tensor: %s. Will retry", e.what());
                return;
            }

            if(outputBufferSize == 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate output tensor: %s. Will retry", this->inferencer->get_error_string());
                return;
            }
            
            if(outputBufferSize != sizeof(float)) {
                RCLCPP_FATAL(this->get_logger(), "Invaild byte size of output tensor. Need 4 but get %ld. Exiting ", outputBufferSize);
                rclcpp::shutdown();
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully allocate output tensor %s", this->output_name_.c_str());
            this->status = Status::ALLOC_INPUT;
        }
        case Status::ALLOC_INPUT:
            try {
                inputBufferSize = this->inferencer->get_input_buffer(input_name_, &inputBuffer);
            }
            catch(const std::runtime_error& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate input tensor %s: %s. Will retry", this->input_name_.c_str(), e.what());
                return;
            }

            if(inputBufferSize == 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate input tensor %s: %s. Will retry", this->input_name_.c_str(), this->inferencer->get_error_string());
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully allocate input tensor %s", this->input_name_.c_str());

            image_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "image_raw", 10, std::bind(&NNControllerNode::initImageCallback, this, std::placeholders::_1));

            this->status = Status::WAIT_IMAGE;
        default:
            retry_init_timer->cancel();
            break;
        }
    }

    void initImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        this->declare_parameter<int>("roi/width", 640);
        this->declare_parameter<int>("roi/height", 480);
    }

    void processImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Image processing and inference logic goes here

        if(!this->inferencer->infer()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to run inference %s", this->inferencer->get_error_string().c_str());
            image_sub.shutdown();
            retry_init_timer->reset();
            status = Status::INIT_BACKEND;
            return;
        }
    }
}

// Entry point
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NNControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}