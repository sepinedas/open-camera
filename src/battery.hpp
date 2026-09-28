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
    double volts = 0.0;    // pack terminal voltage (INA219 bus/load side)
    double amps = 0.0;     // + into the pack (charging), - out of it
    double watts = 0.0;
    int percent = 0;       // 0..100, derived from the smoothed voltage
    bool charging = false; // more than ~50 mA flowing into the pack
};

// Per-model constants for one Waveshare UPS HAT: I2C addresses, the INA219
// calibration its shunt needs, and the pack's voltage curve. Defined in
// battery.cpp, where the table of supported boards lives.
struct BatteryBoard;

// Battery gauge on a Waveshare UPS HAT.
//
// Both supported models carry an INA219 shunt monitor on I2C reporting the
// pack's voltage, current and power -- the UPS HAT (B) at 0x42 with two 18650
// cells in series, the UPS HAT (D) at 0x43 with a single 21700 cell and an
// extra power-path MCU at 0x2D. Everything that differs between them lives in
// the board table; registers, calibration and the state-of-charge curve all
// follow Waveshare's own reference driver for each board, so the numbers match
// what their `INA219.py` demos print.
//
// Reading is a handful of register transfers every couple of seconds, so it
// runs inline on the render thread -- poll() is cheap to call every frame and
// only touches the bus once its interval has elapsed.
class Battery {
public:
    // Opens the gauge on /dev/i2c-<cfg.batteryBus>, probing for the model named
    // by cfg.batteryHat (or for either, on Auto). Returns nullptr when the
    // monitor is disabled, the bus is missing/inaccessible, or no HAT answers
    // -- in every case the app just runs without a gauge.
    static std::unique_ptr<Battery> open(const Config& cfg);

    ~Battery();

    Battery(const Battery&) = delete;
    Battery& operator=(const Battery&) = delete;

    // Re-read the gauge if the poll interval has elapsed; otherwise a no-op.
    void poll();

    const BatteryStatus& status() const { return st_; }

    // True once the pack has stayed under the cut-off voltage, off charge, for
    // the whole grace period. Only acted on with --battery-shutdown.
    bool criticallyLow() const { return critical_; }

    // Ask the HAT to power the Pi back up once the pack recovers, so a
    // low-battery shutdown isn't a dead end. False when this model has no
    // power-path MCU to ask (the UPS HAT (B)) or it didn't answer.
    bool armAutoRestart();

    // Human-readable source, e.g. "Waveshare UPS HAT (B) @ /dev/i2c-1 0x42".
    const std::string& description() const { return desc_; }

private:
    using Clock = std::chrono::steady_clock;

    Battery(int fd, bool shutdown);

    // 16-bit big-endian register access on `addr` (the INA219 unless stated).
    bool readReg(std::uint8_t addr, std::uint8_t reg, std::uint16_t& val) const;
    bool writeReg(std::uint8_t addr, std::uint8_t reg, std::uint16_t val) const;

    // Does `board` answer on this bus? If so, load its calibration/config into
    // the INA219 and adopt it. Leaves board_ untouched on failure.
    bool adopt(const BatteryBoard& board);

    // One full measurement: voltage, current, power, smoothed state-of-charge.
    bool sample();

    int fd_ = -1;
    const BatteryBoard* board_ = nullptr; // the model that answered
    std::string desc_;
    bool shutdownEnabled_ = false;

    BatteryStatus st_;
    double smoothedV_ = 0.0;   // EMA of the bus voltage (load sags are spiky)
    bool critical_ = false;
    Clock::time_point lastPoll_{};
    Clock::time_point lowSince_{}; // when the pack first dropped under cut-off
    bool low_ = false;
};

} // namespace olc
