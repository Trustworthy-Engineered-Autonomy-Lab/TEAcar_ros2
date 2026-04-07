from smbus2 import SMBus
import time

BUS_ID = 7
MPU_ADDR = 0x68

PWR_MGMT_1 = 0x6B

ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F

def read_word_2c(bus, addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value = -((65535 - value) + 1)
    return value

def main():
    bus = SMBus(BUS_ID)

    # 唤醒 MPU6050
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

    print("Reading ACCEL data...\n")

    while True:
        ax_raw = read_word_2c(bus, MPU_ADDR, ACCEL_XOUT_H)
        ay_raw = read_word_2c(bus, MPU_ADDR, ACCEL_YOUT_H)
        az_raw = read_word_2c(bus, MPU_ADDR, ACCEL_ZOUT_H)

        g = 9.80665

        ax = (ax_raw / 16384.0) * g
        ay = (ay_raw / 16384.0) * g
        az = (az_raw / 16384.0) * g

        print(f"ax = {ax:6.2f} m/s² | ay = {ay:6.2f} m/s² | az = {az:6.2f} m/s²")
        time.sleep(0.2)

if __name__ == "__main__":
    main()