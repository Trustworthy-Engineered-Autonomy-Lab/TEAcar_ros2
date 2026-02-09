#include "controller/controller.hpp"
#include "inferencer/inferencer.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <chrono>
#include <cv_bridge/cv_bridge.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

class NNControllerNode : public controller::Controller
{
public:
    NNControllerNode()
        : controller::Controller("nn_controller_node"),
          status_(Status::INIT_BACKEND),
          inferencer_(nullptr)
    {

        // Define default parameters
        this->declare_parameter<std::string>("backend", "tensorflow");
        this->declare_parameter<std::string>("model_file", "");
        this->declare_parameter<std::string>("output_name", "outputs");
        this->declare_parameter<std::string>("input_name", "inputs");

        this->declare_parameter<int>("roi/x", 0);
        this->declare_parameter<int>("roi/y", 0);
        this->declare_parameter<int>("roi/width", 0);  // 0 => default to image width
        this->declare_parameter<int>("roi/height", 0); // 0 => default to image height

        // Load parameters once at startup
        backend_ = this->get_parameter("backend").as_string();
        model_file_ = this->get_parameter("model_file").as_string();
        output_name_ = this->get_parameter("output_name").as_string();
        input_name_ = this->get_parameter("input_name").as_string();

        constexpr auto interval = std::chrono::seconds(1);
        retry_init_timer_ = this->create_wall_timer(
            interval, std::bind(&NNControllerNode::initCallback, this));

        RCLCPP_INFO(this->get_logger(), "NNControllerNode initialization started.");
    }

private:
    // Parameters
    std::string backend_;
    std::string model_file_;
    std::string output_name_;
    std::string input_name_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::TimerBase::SharedPtr retry_init_timer_;

    cv::Rect roi_;
    cv::Mat image_mat_; // wraps input_buffer_ as float32

    enum class Status
    {
        INIT_BACKEND,
        LOAD_MODEL,
        ALLOC_OUTPUT,
        ALLOC_INPUT,
        WAIT_FIRST_IMAGE,
        RUNNING,
    };

    Status status_;
    std::shared_ptr<inferencer::Inferencer> inferencer_;

    void *output_buffer_ = nullptr;
    void *input_buffer_ = nullptr;
    size_t output_buffer_size_ = 0;
    size_t input_buffer_size_ = 0;

    void resetToInit()
    {
        inferencer_.reset();
        output_buffer_ = nullptr;
        input_buffer_ = nullptr;
        output_buffer_size_ = 0;
        input_buffer_size_ = 0;
        image_mat_.release();
        roi_ = cv::Rect();

        // Drop subscription
        image_sub_.reset();

        status_ = Status::INIT_BACKEND;

        // Restart timer if it was cancelled
        if (retry_init_timer_)
        {
            retry_init_timer_->reset();
        }
    }

    void initCallback()
    {
        switch (status_)
        {
        case Status::INIT_BACKEND:
        {
            try
            {
                // Dynamically load the inferencer backend
                inferencer_ = std::make_shared<inferencer::Inferencer>(backend_);
            }
            catch (const std::runtime_error &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize the %s backend: %s. Will retry",
                             backend_.c_str(), e.what());
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully initialized %s backend", backend_.c_str());
            status_ = Status::LOAD_MODEL;
            [[fallthrough]];
        }

        case Status::LOAD_MODEL:
        {
            if (model_file_.empty())
            {
                // TODO: default model file path logic
                RCLCPP_ERROR(this->get_logger(), "Model file parameter is empty. Will retry");
                return;
            }

            bool ok = false;
            try
            {
                ok = inferencer_->load_model(model_file_.c_str());
            }
            catch (const std::runtime_error &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to load model file %s: %s. Will retry",
                             model_file_.c_str(), e.what());
                return;
            }

            if (!ok)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to load model file %s: %s. Will retry",
                             model_file_.c_str(), inferencer_->get_error_string());
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully loaded model file %s", model_file_.c_str());
            status_ = Status::ALLOC_OUTPUT;
            [[fallthrough]];
        }

        case Status::ALLOC_OUTPUT:
        {
            try
            {
                output_buffer_size_ = inferencer_->get_output_buffer(output_name_.c_str(), &output_buffer_);
            }
            catch (const std::runtime_error &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to alloc output '%s': %s. Will retry.",
                             output_name_.c_str(), e.what());
                return;
            }

            if (output_buffer_size_ == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to alloc output '%s': %s. Will retry.",
                             output_name_.c_str(), inferencer_->get_error_string());
                return;
            }

            if (output_buffer_size_ != sizeof(float))
            {
                RCLCPP_FATAL(this->get_logger(),
                             "Invalid byte size of output tensor. Need %zu but got %zu. Exiting",
                             sizeof(float), output_buffer_size_);
                rclcpp::shutdown();
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Successfully allocated output tensor %s", output_name_.c_str());
            status_ = Status::ALLOC_INPUT;
            [[fallthrough]];
        }

        case Status::ALLOC_INPUT:
        {
            try
            {
                input_buffer_size_ = inferencer_->get_input_buffer(input_name_.c_str(), &input_buffer_);
            }
            catch (const std::runtime_error &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate input tensor %s: %s. Will retry",
                             input_name_.c_str(), e.what());
                return;
            }

            if (input_buffer_size_ == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate input tensor %s: %s. Will retry",
                             input_name_.c_str(), inferencer_->get_error_string());
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Allocated input '%s' (%zu bytes).", input_name_.c_str(),
                        input_buffer_size_);

            // Subscribe for first image to lock ROI + create cv::Mat wrapper
            image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "image_raw", rclcpp::SensorDataQoS(),
                std::bind(&NNControllerNode::firstImageCallback, this, std::placeholders::_1));

            status_ = Status::WAIT_FIRST_IMAGE;
            [[fallthrough]];
        }

        case Status::WAIT_FIRST_IMAGE:
        case Status::RUNNING:
        default:
            // Once we are waiting for images, stop the retry timer.
            if (retry_init_timer_)
                retry_init_timer_->cancel();
            return;
        }
    }

    void firstImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        const int x = this->get_parameter("roi/x").as_int();
        const int y = this->get_parameter("roi/y").as_int();

        int roi_w = this->get_parameter("roi/width").as_int();
        int roi_h = this->get_parameter("roi/height").as_int();
        if (roi_w <= 0)
            roi_w = static_cast<int>(msg->width);
        if (roi_h <= 0)
            roi_h = static_cast<int>(msg->height);

        const cv::Rect valid_region(0, 0, static_cast<int>(msg->width), static_cast<int>(msg->height));
        roi_ = cv::Rect(x, y, roi_w, roi_h) & valid_region;

        int channels = 0;
        if (msg->encoding == sensor_msgs::image_encodings::MONO8)
        {
            channels = 1;
        }
        else if (msg->encoding == sensor_msgs::image_encodings::RGB8)
        {
            channels = 3;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Unsupported image encoding '%s'. Will retry.",
                         msg->encoding.c_str());
            return;
        }

        const size_t roi_byte_size =
            static_cast<size_t>(roi_.area()) * sizeof(float) * static_cast<size_t>(channels);
        if (roi_byte_size != input_buffer_size_)
        {
            RCLCPP_ERROR(this->get_logger(), "ROI byte size %zu != input buffer size %zu. Will retry.",
                         roi_byte_size, input_buffer_size_);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Valid ROI: x=%d y=%d w=%d h=%d (channels=%d)", roi_.x, roi_.y,
                    roi_.width, roi_.height, channels);

        // Wrap input buffer as float image
        image_mat_ =
            cv::Mat(roi_.height, roi_.width, (channels == 1) ? CV_32FC1 : CV_32FC3, input_buffer_);

        // Switch to running callback
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_raw", rclcpp::SensorDataQoS(),
            std::bind(&NNControllerNode::runningImageCallback, this, std::placeholders::_1));

        status_ = Status::RUNNING;
    }

    void runningImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImageConstPtr image_ptr;
        try
        {
            image_ptr = cv_bridge::toCvShare(msg, msg->encoding);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
            return;
        }

        // Copy ROI into input buffer as float32
        image_ptr->image(roi_).convertTo(image_mat_,
                                         (image_ptr->image.channels() == 1) ? CV_32FC1 : CV_32FC3);

        if (!inferencer_->infer())
        {
            RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", inferencer_->get_error_string());
            // unsubscribe and re-init
            resetToInit();
            return;
        }

        const float steer = *reinterpret_cast<float *>(output_buffer_);

        // Only controls steer; throttle is managed elsewhere.
        // Note: original code used control(0.0f, steer). If your base class does not
        // provide controlSteer(), change this call back to control(0.0f, steer).
        this->control(0.0f, steer); // publishes /motion_cmd

        RCLCPP_DEBUG(this->get_logger(), "Inference OK, steer=%f", steer);
    }
};

// Entry point
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NNControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}