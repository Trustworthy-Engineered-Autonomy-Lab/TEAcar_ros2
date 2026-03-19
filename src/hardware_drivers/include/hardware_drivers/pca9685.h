#ifndef PCA9685_H
#define PCA9685_H

#include <string>

class PCA9685
{
public:
    PCA9685();
    ~PCA9685();

    bool open(const std::string &device_name, float freq = 60, int address = 0x40);
    bool is_opened();
    void close();
    float get_pwm_freq();
    bool set_pwm_freq(float freq);
    bool set_pwm_dutycycle(int ch, float duty_cycle);
    std::string get_error();
    bool reset();
    bool wakeup();
    bool sleep();

private:
    int fd_ = -1;
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
};


#endif