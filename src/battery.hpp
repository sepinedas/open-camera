#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "config.hpp"

namespace olc {

// A snapshot of the pack, refreshed by Battery::poll().
struct BatteryStatus {
    bool valid = false;    // at least one reading has succeeded
    double volts = 0.0;    // cell terminal voltage (INA219 bus/load side)
    double amps = 0.0;     // + into the cell (charging), - out of it
    double watts = 0.0;
    int percent = 0;       // 0..100, derived from the smoothed voltage
    bool charging = false; // more than ~50 mA flowing into the cell
};

// Battery gauge on a Waveshare UPS HAT (D).
//
// The HAT carries an INA219 shunt monitor on I2C (address 0x43) reporting the
// cell's voltage, current and power, plus a small MCU (0x2D) that owns the
// power path. Registers, calibration and the state-of-charge curve all follow
// Waveshare's own reference driver for this board, so the numbers match what
// their `INA219.py` demo prints.
//
// Reading is a handful of register transfers every couple of seconds, so it
// runs inline on the render thread -- poll() is cheap to call every frame and
// only touches the bus once its interval has elapsed.
class Battery {
public:
    // Opens the gauge on /dev/i2c-<cfg.batteryBus>. Returns nullptr when the
    // monitor is disabled, the bus is missing/inaccessible, or nothing answers
    // at the HAT's address -- in every case the app just runs without a gauge.
    static std::unique_ptr<Battery> open(const Config& cfg);

    ~Battery();

    Battery(const Battery&) = delete;
    Battery& operator=(const Battery&) = delete;

    // Re-read the gauge if the poll interval has elapsed; otherwise a no-op.
    void poll();

    const BatteryStatus& status() const { return st_; }

    // True once the cell has stayed under the cut-off voltage, off charge, for
    // the whole grace period. Only acted on with --battery-shutdown.
    bool criticallyLow() const { return critical_; }

    // Ask the HAT's MCU to power the Pi back up once the cell recovers, so a
    // low-battery shutdown isn't a dead end. False if the MCU didn't answer.
    bool armAutoRestart();

    // Human-readable source, e.g. "Waveshare UPS HAT (D) @ /dev/i2c-1 0x43".
    const std::string& description() const { return desc_; }

private:
    using Clock = std::chrono::steady_clock;

    Battery(int fd, std::string desc, bool shutdown);

    // 16-bit big-endian register access on `addr` (INA219 unless stated).
    bool readReg(std::uint8_t addr, std::uint8_t reg, std::uint16_t& val) const;
    bool writeReg(std::uint8_t addr, std::uint8_t reg, std::uint16_t val) const;

    // One full measurement: voltage, current, power, smoothed state-of-charge.
    bool sample();

    int fd_ = -1;
    std::string desc_;
    bool shutdownEnabled_ = false;

    BatteryStatus st_;
    double smoothedV_ = 0.0;   // EMA of the bus voltage (load sags are spiky)
    bool critical_ = false;
    Clock::time_point lastPoll_{};
    Clock::time_point lowSince_{}; // when the cell first dropped under cut-off
    bool low_ = false;
};

} // namespace olc
