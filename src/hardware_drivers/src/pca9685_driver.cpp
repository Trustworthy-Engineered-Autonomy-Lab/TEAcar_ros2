// pca9685_actuator_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <memory>
#include <cmath>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

class PCA9685
{
public:
  PCA9685() = default;
  ~PCA9685() { close(); }

  bool open(const std::string &device_name, float freq = 60, int address = 0x40)
  {
    fd_ = ::open(device_name.c_str(), O_RDWR);
    if (fd_ < 0)
    {
      error_ = "Failed to open I2C device: " + device_name;
      return false;
    }
    if (ioctl(fd_, I2C_SLAVE, address) < 0)
    {
      error_ = "Failed to get I2C access for: " + device_name;
      goto clean_up;
    }

    if (!(reset() && set_pwm_freq(freq)))
      goto clean_up;

    return true;
  clean_up:
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  bool is_opened()
  {
    return fd_ >= 0;
  }

  void close()
  {
    if (fd_ >= 0)
    {
      sleep();
      ::close(fd_);
      fd_ = -1;
    }
  }

  float get_pwm_freq()
  {
    uint8_t prescale;
    if (!read_reg(PRESCALE, &prescale))
      return 0;

    return oscillator_freq / (4096.0 * (prescale + 1));
  }

  bool set_pwm_freq(float freq)
  {
    // Range output modulation frequency is dependant on oscillator
    if (freq < 1)
      freq = 1;
    if (freq > 3500)
      freq = 3500; // Datasheet limit is 3052=50MHz/(4*4096)

    float prescaleval = ((oscillator_freq / (freq * 4096.0)) + 0.5) - 1;
    if (prescaleval < PCA9685_PRESCALE_MIN)
      prescaleval = PCA9685_PRESCALE_MIN;
    if (prescaleval > PCA9685_PRESCALE_MAX)
      prescaleval = PCA9685_PRESCALE_MAX;
    uint8_t prescale = static_cast<uint8_t>(prescaleval);

    uint8_t oldmode;
    if (!read_reg(MODE1, &oldmode))
      return false;

    // sleep and set prescaler
    uint8_t newmode = (oldmode & ~MODE1_RESTART) | MODE1_SLEEP;
    bool success = write_reg(MODE1, newmode) && write_reg(PRESCALE, prescale) && write_reg(MODE1, oldmode);
    if (!success)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    // This sets the MODE1 register to turn on auto increment.
    if (!write_reg(MODE1, oldmode | MODE1_RESTART | MODE1_AI))
      return 0;

    return true;
  }

  bool set_pwm_dutycycle(int ch, float duty_cycle)
  {
    uint16_t off = static_cast<uint16_t>(4096 * duty_cycle);
    return write_reg(LED0_ON_L + 4 * ch, 0) &&
           write_reg(LED0_ON_H + 4 * ch, 0) &&
           write_reg(LED0_OFF_L + 4 * ch, off & 0xFF) &&
           write_reg(LED0_OFF_H + 4 * ch, off >> 8);
  }

  std::string get_error()
  {
    return std::strerror(errno);
  }

private:
  int fd_ = -1;
  std::string error_;
  static constexpr uint8_t MODE1 = 0x00, PRESCALE = 0xFE;
  static constexpr uint8_t LED0_ON_L = 0x06, LED0_ON_H = 0x07;
  static constexpr uint8_t LED0_OFF_L = 0x08, LED0_OFF_H = 0x09;
  static constexpr uint8_t ALL_LED_OFF_H = 0xFD;

  static constexpr uint8_t MODE1_RESTART = 0x80;
  static constexpr uint8_t MODE1_SLEEP = 0x10;
  static constexpr uint8_t MODE1_AI = 0x20;

  static constexpr float oscillator_freq = 25000000.0f;
  static constexpr uint8_t PCA9685_PRESCALE_MIN = 3;
  static constexpr uint8_t PCA9685_PRESCALE_MAX = 255;

  bool reset()
  {
    return write_reg(MODE1, MODE1_RESTART);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  bool wakeup()
  {
    uint8_t sleep;
    if (!read_reg(MODE1, &sleep))
      return false;
    uint8_t wakeup = sleep & ~MODE1_SLEEP;
    return write_reg(MODE1, wakeup);
  }

  bool sleep()
  {
    uint8_t awake;
    if (!read_reg(MODE1, &awake))
      return false;
    uint8_t sleep = awake | MODE1_SLEEP;
    if (!write_reg(MODE1, sleep))
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return true;
  }

  bool write_reg(uint8_t reg, uint8_t val)
  {
    uint8_t buf[2] = {reg, val};
    return write(fd_, buf, 2) == 2;
  }

  bool read_reg(uint8_t reg, uint8_t *val)
  {
    return write(fd_, &reg, 1) == 1 && read(fd_, val, 1) == 1;
  }
};

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
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to open pca9685 on %s. Will retry", bus_device_.c_str());
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