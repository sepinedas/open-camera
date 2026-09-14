#include "battery.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef __linux__
#include <fcntl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace olc {

namespace {

// --- Waveshare UPS HAT (D) --------------------------------------------------
// The addresses and constants below come from Waveshare's reference driver for
// this exact board (UPS_HAT_D/INA219.py). The (B) and (C) HATs use different
// shunt resistors and calibration values, so they are not interchangeable.

constexpr std::uint8_t kInaAddr = 0x43; // INA219 shunt/bus monitor
constexpr std::uint8_t kMcuAddr = 0x2D; // on-board power-path MCU

// INA219 register map.
constexpr std::uint8_t kRegConfig = 0x00;
constexpr std::uint8_t kRegShunt = 0x01;
constexpr std::uint8_t kRegBus = 0x02;
constexpr std::uint8_t kRegPower = 0x03;
constexpr std::uint8_t kRegCurrent = 0x04;
constexpr std::uint8_t kRegCal = 0x05;

// 16 V bus range, /2 gain (80 mV shunt), 12-bit 32-sample averaging on both
// ADCs, shunt+bus continuous:
//   RANGE_16V << 13 | DIV_2_80MV << 11 | 12BIT_32S << 7 | 12BIT_32S << 3 | 7
constexpr std::uint16_t kConfig = 0x0EEF;

// The HAT's shunt is 0.01 ohm, sized for the board's 5 A output:
//   cal = trunc(0.04096 / (current_lsb * 0.01)) with current_lsb = 152.4 uA.
constexpr std::uint16_t kCalibration = 26868;
constexpr double kCurrentLsbA = 0.0001524; // A per current-register bit
constexpr double kPowerLsbW = 0.003048;    // W per power-register bit
constexpr double kBusLsbV = 0.004;         // V per bus-register bit (15..3)

// State of charge for the single 21700 Li-ion cell, straight off the terminal
// voltage: 3.0 V is treated as empty and 4.2 V as full.
constexpr double kEmptyV = 3.0;
constexpr double kFullV = 4.2;

// Current above which the cell counts as charging rather than merely idle.
constexpr double kChargingA = 0.05;

// A connected cell never reads anywhere near this low -- protection would have
// cut it off long before. Anything under it means the first conversion has not
// landed yet (the ADC averages 32 samples) or no pack is fitted, neither of
// which should be shown as a flat battery.
constexpr double kPresentV = 2.0;

// Below this voltage, off charge, the pack is nearly flat. Waveshare's demo
// shuts the Pi down at the same threshold after a minute of it.
constexpr double kCutoffV = 3.15;
constexpr auto kCutoffGrace = std::chrono::seconds(60);

constexpr auto kPollInterval = std::chrono::milliseconds(2000);

// Weight of each new reading in the voltage EMA. The terminal voltage dips
// whenever the camera or the panel draws a burst, so an unsmoothed percentage
// jitters by several points from frame to frame.
constexpr double kSmoothing = 0.25;

// Reinterpret a raw register as the two's-complement value the INA219 reports
// for its signed shunt/current/power registers.
inline int signed16(std::uint16_t raw) {
    return (raw > 32767) ? (int)raw - 65536 : (int)raw;
}

} // namespace

Battery::Battery(int fd, std::string desc, bool shutdown)
    : fd_(fd), desc_(std::move(desc)), shutdownEnabled_(shutdown) {}

Battery::~Battery() {
#ifdef __linux__
    if (fd_ >= 0) ::close(fd_);
#endif
}

#ifdef __linux__

// Both accessors drive a single I2C transaction through I2C_RDWR, so the read
// gets a proper repeated start and one file descriptor can address both the
// INA219 and the MCU without re-binding a slave address between calls.
bool Battery::readReg(std::uint8_t addr, std::uint8_t reg,
                      std::uint16_t& val) const {
    std::uint8_t out[2] = {0, 0};
    i2c_msg msgs[2]{};
    msgs[0].addr = addr;
    msgs[0].flags = 0;
    msgs[0].len = 1;
    msgs[0].buf = &reg;
    msgs[1].addr = addr;
    msgs[1].flags = I2C_M_RD;
    msgs[1].len = 2;
    msgs[1].buf = out;

    i2c_rdwr_ioctl_data xfer{};
    xfer.msgs = msgs;
    xfer.nmsgs = 2;
    if (::ioctl(fd_, I2C_RDWR, &xfer) < 0) return false;

    val = (std::uint16_t)((out[0] << 8) | out[1]); // big-endian on the wire
    return true;
}

bool Battery::writeReg(std::uint8_t addr, std::uint8_t reg,
                       std::uint16_t val) const {
    std::uint8_t buf[3] = {reg, (std::uint8_t)(val >> 8),
                           (std::uint8_t)(val & 0xFF)};
    i2c_msg msg{};
    msg.addr = addr;
    msg.flags = 0;
    msg.len = 3;
    msg.buf = buf;

    i2c_rdwr_ioctl_data xfer{};
    xfer.msgs = &msg;
    xfer.nmsgs = 1;
    return ::ioctl(fd_, I2C_RDWR, &xfer) >= 0;
}

std::unique_ptr<Battery> Battery::open(const Config& cfg) {
    if (!cfg.battery) return nullptr;

    std::string path = "/dev/i2c-" + std::to_string(cfg.batteryBus);
    int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
        // Not worth shouting about -- plenty of machines have no I2C at all --
        // but say enough to fix it if a HAT *was* expected.
        if (errno == ENOENT)
            std::cerr << "battery: " << path << " not present; running without "
                         "a gauge (enable I2C with dtparam=i2c_arm=on)\n";
        else
            std::cerr << "battery: cannot open " << path << ": "
                      << std::strerror(errno)
                      << " (add your user to the i2c group)\n";
        return nullptr;
    }

    std::unique_ptr<Battery> b(new Battery(fd, path, cfg.batteryShutdown));

    // Probe: anything at the HAT's address will ACK a config read.
    std::uint16_t probe = 0;
    if (!b->readReg(kInaAddr, kRegConfig, probe)) {
        std::cerr << "battery: no UPS HAT (D) at 0x43 on " << path
                  << "; running without a gauge\n";
        return nullptr;
    }

    if (!b->writeReg(kInaAddr, kRegCal, kCalibration) ||
        !b->writeReg(kInaAddr, kRegConfig, kConfig)) {
        std::cerr << "battery: UPS HAT (D) found but would not accept its "
                     "configuration; running without a gauge\n";
        return nullptr;
    }

    b->desc_ = "Waveshare UPS HAT (D) @ " + path + " 0x43";
    b->sample(); // seed the smoothed voltage so the first frame shows a level
    b->lastPoll_ = Clock::now();
    return b;
}

bool Battery::sample() {
    // The INA219 loses its calibration on a bus glitch or a brown-out, which
    // silently zeroes the current and power registers. Waveshare's driver
    // rewrites it before every read; do the same rather than trust it to stick.
    if (!writeReg(kInaAddr, kRegCal, kCalibration)) return false;

    std::uint16_t bus = 0, shunt = 0, current = 0, power = 0;
    if (!readReg(kInaAddr, kRegBus, bus) ||
        !readReg(kInaAddr, kRegShunt, shunt) ||
        !readReg(kInaAddr, kRegCurrent, current) ||
        !readReg(kInaAddr, kRegPower, power))
        return false;

    (void)shunt; // read to keep the sequence identical to the reference driver

    // Bus register: bits 15..3 hold the voltage, the low bits are status flags.
    double volts = (double)(bus >> 3) * kBusLsbV;
    if (volts < kPresentV) {
        st_.valid = false; // no pack, or the ADC hasn't produced a reading yet
        return false;
    }

    // The shunt is wired so discharge current reads positive on this board;
    // flip it to the convention used everywhere else here (+ = into the cell),
    // matching the negation in Waveshare's demo.
    double amps = -(double)signed16(current) * kCurrentLsbA;
    double watts = (double)signed16(power) * kPowerLsbW;

    smoothedV_ = (smoothedV_ <= 0.0)
                     ? volts
                     : smoothedV_ + kSmoothing * (volts - smoothedV_);

    double pct = (smoothedV_ - kEmptyV) / (kFullV - kEmptyV) * 100.0;

    st_.valid = true;
    st_.volts = volts;
    st_.amps = amps;
    st_.watts = watts;
    st_.percent = (int)std::lround(std::clamp(pct, 0.0, 100.0));
    st_.charging = amps > kChargingA;
    return true;
}

bool Battery::armAutoRestart() {
    // Register 0x01 of the HAT's MCU: 0x55 makes it power the Pi back up by
    // itself once the cell has recovered, instead of staying off until someone
    // presses the button. It is a single-byte write, so not writeReg().
    std::uint8_t buf[2] = {0x01, 0x55};
    i2c_msg msg{};
    msg.addr = kMcuAddr;
    msg.flags = 0;
    msg.len = 2;
    msg.buf = buf;

    i2c_rdwr_ioctl_data xfer{};
    xfer.msgs = &msg;
    xfer.nmsgs = 1;
    return ::ioctl(fd_, I2C_RDWR, &xfer) >= 0;
}

#else // !__linux__

// I2C through ioctl is Linux-only; elsewhere the app runs without a gauge.
bool Battery::readReg(std::uint8_t, std::uint8_t, std::uint16_t&) const { return false; }
bool Battery::writeReg(std::uint8_t, std::uint8_t, std::uint16_t) const { return false; }
bool Battery::sample() { return false; }
bool Battery::armAutoRestart() { return false; }

std::unique_ptr<Battery> Battery::open(const Config&) { return nullptr; }

#endif // __linux__

void Battery::poll() {
    Clock::time_point now = Clock::now();
    if (now - lastPoll_ < kPollInterval) return;
    lastPoll_ = now;

    // A transient bus error shouldn't blank the badge: keep the last good
    // reading and try again on the next interval.
    if (!sample()) return;

    // Track how long the cell has been under the cut-off while off charge. A
    // single dip during a capture burst is normal, so only a sustained low
    // reading counts as critical.
    const bool under = st_.volts < kCutoffV && !st_.charging;
    if (under) {
        if (!low_) {
            low_ = true;
            lowSince_ = now;
        }
    } else {
        low_ = false;
    }
    critical_ = low_ && (now - lowSince_) >= kCutoffGrace;

    if (low_ && !critical_ && shutdownEnabled_) {
        auto left = std::chrono::duration_cast<std::chrono::seconds>(
                        kCutoffGrace - (now - lowSince_))
                        .count();
        std::cerr << "battery: " << st_.volts << " V is below the " << kCutoffV
                  << " V cut-off; powering off in " << left
                  << " s unless charged\n";
    }
}

} // namespace olc
