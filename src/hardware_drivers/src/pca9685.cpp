#include "hardware_drivers/pca9685.h"

extern "C"
{
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <i2c/smbus.h>
}

#include <string>
#include <chrono>
#include <thread>

#include <cerrno>
#include <cstring>

PCA9685::PCA9685()
{
}

PCA9685::~PCA9685()
{
    close();
}

bool PCA9685::open(const std::string &device_name, float freq, int address)
{
    fd_ = ::open(device_name.c_str(), O_RDWR);
    if (fd_ < 0)
    {
        return false;
    }
    if (ioctl(fd_, I2C_SLAVE, address) < 0)
    {
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

bool PCA9685::is_opened()
{
    return fd_ >= 0;
}

void PCA9685::close()
{
    if (fd_ >= 0)
    {
        sleep();
        ::close(fd_);
        fd_ = -1;
    }
}

float PCA9685::get_pwm_freq()
{
    int result = i2c_smbus_read_byte_data(fd_, PRESCALE);
    if (result < 0)
        return 0;

    uint8_t prescale = result & 0xFF;

    return oscillator_freq / (4096.0 * (prescale + 1));
}

bool PCA9685::set_pwm_freq(float freq)
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

    int result = i2c_smbus_read_byte_data(fd_, MODE1);
    if (result < 0)
        return false;

    uint8_t oldmode = result & 0xFF;

    // sleep and set prescaler
    uint8_t newmode = (oldmode & ~MODE1_RESTART) | MODE1_SLEEP;
    bool success = i2c_smbus_write_byte_data(fd_, MODE1, newmode) >= 0 &&
                   i2c_smbus_write_byte_data(fd_, PRESCALE, prescale) >= 0 &&
                   i2c_smbus_write_byte_data(fd_, MODE1, oldmode) >= 0;
    if (!success)
        return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    // This sets the MODE1 register to turn on auto increment.
    if (i2c_smbus_write_byte_data(fd_, MODE1, oldmode | MODE1_RESTART | MODE1_AI) < 0)
        return false;

    return true;
}

bool PCA9685::set_pwm_dutycycle(int ch, float duty_cycle)
{
    uint16_t off = static_cast<uint16_t>(4096 * duty_cycle);
    return i2c_smbus_write_word_data(fd_, LED0_ON_L + 4 * ch, 0) >= 0 &&
           i2c_smbus_write_word_data(fd_, LED0_OFF_L + 4 * ch, off) >= 0;
}

std::string PCA9685::get_error()
{
    return std::strerror(errno);
}

bool PCA9685::reset()
{
    int result = i2c_smbus_write_byte_data(fd_, MODE1, MODE1_RESTART);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return result >= 0;
}

bool PCA9685::wakeup()
{
    int result = i2c_smbus_read_byte_data(fd_, MODE1);
    if (result < 0)
        return false;

    uint8_t sleep = result & 0xFF;
    uint8_t wakeup = sleep & ~MODE1_SLEEP;
    return i2c_smbus_write_byte_data(fd_, MODE1, wakeup) >= 0;
}

bool PCA9685::sleep()
{
    int result = i2c_smbus_read_byte_data(fd_, MODE1);
    if (result < 0)
        return false;

    uint8_t awake = result & 0xFF;
    uint8_t sleep = awake | MODE1_SLEEP;
    if (i2c_smbus_write_byte_data(fd_, MODE1, sleep) < 0)
        return false;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return true;
}
