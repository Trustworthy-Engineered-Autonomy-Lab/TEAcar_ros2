#include "hardware_drivers/mpu6050.h"

#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

using namespace std::chrono_literals;

class MPU6050DriverNode : public rclcpp::Node
{
public:
    MPU6050DriverNode()
        : Node("mpu6050publisher"), mpu6050_{std::make_unique<MPU6050>()}
    {
        // Declare parameters
        declareParameters();
        // Set parameters
        mpu6050_->setGyroscopeRange(
            static_cast<MPU6050::GyroRange>(this->get_parameter("gyro_range").as_int()));
        mpu6050_->setAccelerometerRange(
            static_cast<MPU6050::AccelRange>(this->get_parameter("accel_range").as_int()));
        mpu6050_->setDlpfBandwidth(
            static_cast<MPU6050::DlpfBandwidth>(this->get_parameter("dlpf_bandwidth").as_int()));
        mpu6050_->setGyroscopeOffset(this->get_parameter("gyro_x_offset").as_double(),
                                     this->get_parameter("gyro_y_offset").as_double(),
                                     this->get_parameter("gyro_z_offset").as_double());
        mpu6050_->setAccelerometerOffset(this->get_parameter("accel_x_offset").as_double(),
                                         this->get_parameter("accel_y_offset").as_double(),
                                         this->get_parameter("accel_z_offset").as_double());
        // Check if we want to calibrate the sensor
        if (this->get_parameter("calibrate").as_bool())
        {
            RCLCPP_INFO(this->get_logger(), "Calibrating...");
            mpu6050_->calibrate();
        }
        mpu6050_->printConfig();
        mpu6050_->printOffsets();
        // Create publisher
        publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
        std::chrono::duration<int64_t, std::milli> frequency =
            1000ms / this->get_parameter("gyro_range").as_int();
        timer_ = this->create_wall_timer(frequency, std::bind(&MPU6050DriverNode::handleInput, this));
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
    std::unique_ptr<MPU6050> mpu6050_;
    size_t count_;
    rclcpp::TimerBase::SharedPtr timer_;
    void handleInput()
    {
        auto message = sensor_msgs::msg::Imu();
        message.header.stamp = this->get_clock()->now();
        message.header.frame_id = "base_link";
        message.linear_acceleration_covariance = {0};
        message.linear_acceleration.x = mpu6050_->getAccelerationX();
        message.linear_acceleration.y = mpu6050_->getAccelerationY();
        message.linear_acceleration.z = mpu6050_->getAccelerationZ();
        message.angular_velocity_covariance[0] = {0};
        message.angular_velocity.x = mpu6050_->getAngularVelocityX();
        message.angular_velocity.y = mpu6050_->getAngularVelocityY();
        message.angular_velocity.z = mpu6050_->getAngularVelocityZ();
        // Invalidate quaternion
        message.orientation_covariance[0] = -1;
        message.orientation.x = 0;
        message.orientation.y = 0;
        message.orientation.z = 0;
        message.orientation.w = 0;
        publisher_->publish(message);
    }

    void declareParameters()
    {
        this->declare_parameter<bool>("calibrate", true);
        this->declare_parameter<int>("gyro_range", MPU6050::GyroRange::GYR_250_DEG_S);
        this->declare_parameter<int>("accel_range", MPU6050::AccelRange::ACC_2_G);
        this->declare_parameter<int>("dlpf_bandwidth", MPU6050::DlpfBandwidth::DLPF_260_HZ);
        this->declare_parameter<double>("gyro_x_offset", 0.0);
        this->declare_parameter<double>("gyro_y_offset", 0.0);
        this->declare_parameter<double>("gyro_z_offset", 0.0);
        this->declare_parameter<double>("accel_x_offset", 0.0);
        this->declare_parameter<double>("accel_y_offset", 0.0);
        this->declare_parameter<double>("accel_z_offset", 0.0);
        this->declare_parameter<int>("frequency", 0.0);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPU6050DriverNode>());
    rclcpp::shutdown();
    return 0;
}