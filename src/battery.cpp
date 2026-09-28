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

// --- Per-model constants ----------------------------------------------------
// Everything that differs between the supported UPS HATs. The values come from
// Waveshare's reference driver for each board (UPS_HAT_B/INA219.py and
// UPS_HAT_D/INA219.py) -- the two boards use different shunts, different packs
// and even opposite current polarity, so nothing here is interchangeable.

struct BatteryBoard {
    UpsHat id;
    const char* name;
    std::uint8_t inaAddr;      // INA219 shunt/bus monitor
    std::uint8_t mcuAddr;      // power-path MCU, or 0 when the board has none
    std::uint16_t config;      // INA219 config register
    std::uint16_t calibration; // INA219 calibration register
    double currentLsbA;        // A per current-register bit
    double powerLsbW;          // W per power-register bit
    bool dischargePositive;    // raw current reads + while discharging
    double emptyV, fullV;      // pack voltage curve, 0% -> 100%
    double presentV;           // below this: no pack, or the ADC hasn't settled
    double cutoffV;            // --battery-shutdown threshold
};

namespace {

// INA219 register map (identical on both boards).
constexpr std::uint8_t kRegConfig = 0x00;
constexpr std::uint8_t kRegShunt = 0x01;
constexpr std::uint8_t kRegBus = 0x02;
constexpr std::uint8_t kRegPower = 0x03;
constexpr std::uint8_t kRegCurrent = 0x04;
constexpr std::uint8_t kRegCal = 0x05;

// Bus-voltage scaling is the same in both the 16 V and 32 V ranges.
constexpr double kBusLsbV = 0.004; // V per bus-register bit (15..3)

// Current above which the pack counts as charging rather than merely idle.
constexpr double kChargingA = 0.05;

// How long the pack must stay under the cut-off, off charge, before the app
// treats it as critical.
constexpr auto kCutoffGrace = std::chrono::seconds(60);

constexpr auto kPollInterval = std::chrono::milliseconds(2000);

// Weight of each new reading in the voltage EMA. The terminal voltage dips
// whenever the camera or the panel draws a burst, so an unsmoothed percentage
// jitters by several points from frame to frame.
constexpr double kSmoothing = 0.25;

// The supported boards, probed in this order when --battery-hat is auto. Their
// INA219s sit at different addresses, so the probe is unambiguous.
constexpr BatteryBoard kBoards[] = {
    // UPS HAT (B): 2x 18650 in series (6.0-8.4 V), 0.1 ohm shunt, 32 V / 2 A
    // profile (cal 4096, 100 uA per bit). Its demo prints the raw current
    // as-is, with negative meaning discharge, so the polarity already matches
    // our convention. There is no power-path MCU on this board, and Waveshare
    // ship no low-voltage shutdown for it -- the 6.3 V cut-off below is ours,
    // picked as the same 3.15 V per cell the (D) uses.
    {UpsHat::B, "UPS HAT (B)", 0x42, 0x00, 0x3EEF, 4096, 0.0001, 0.002, false,
     6.0, 8.4, 4.0, 6.3},

    // UPS HAT (D): one 21700 cell (3.0-4.2 V), 0.01 ohm shunt sized for the
    // board's 5 A output, 16 V / 5 A profile (cal 26868, 152.4 uA per bit).
    // Its demo negates the current register, so discharge reads positive here.
    // 0x2D is the MCU that owns the power path.
    {UpsHat::D, "UPS HAT (D)", 0x43, 0x2D, 0x0EEF, 26868, 0.0001524, 0.003048,
     true, 3.0, 4.2, 2.0, 3.15},
};

// Reinterpret a raw register as the two's-complement value the INA219 reports
// for its signed shunt/current/power registers.
inline int signed16(std::uint16_t raw) {
    return (raw > 32767) ? (int)raw - 65536 : (int)raw;
}

// Two-digit hex, for log lines that name an I2C address.
std::string hex8(std::uint8_t v) {
    static const char* d = "0123456789abcdef";
    return std::string("0x") + d[(v >> 4) & 0xF] + d[v & 0xF];
}

// Is this board one the user asked for?
bool wanted(const BatteryBoard& b, UpsHat choice) {
    return choice == UpsHat::Auto || choice == b.id;
}

} // namespace

Battery::Battery(int fd, bool shutdown) : fd_(fd), shutdownEnabled_(shutdown) {}

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

bool Battery::adopt(const BatteryBoard& board) {
    // Anything at the INA219's address will ACK a config read.
    std::uint16_t probe = 0;
    if (!readReg(board.inaAddr, kRegConfig, probe)) return false;

    if (!writeReg(board.inaAddr, kRegCal, board.calibration) ||
        !writeReg(board.inaAddr, kRegConfig, board.config)) {
        std::cerr << "battery: something answered at " << hex8(board.inaAddr)
                  << " but would not accept the " << board.name
                  << " configuration\n";
        return false;
    }

    board_ = &board;
    return true;
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

    std::unique_ptr<Battery> b(new Battery(fd, cfg.batteryShutdown));

    for (const BatteryBoard& board : kBoards)
        if (wanted(board, cfg.batteryHat) && b->adopt(board)) break;

    if (!b->board_) {
        // Name what was actually looked for, so a wrong --battery-hat or a HAT
        // strapped to a non-default address is obvious from the log alone.
        std::cerr << "battery: no UPS HAT answered on " << path << " (tried";
        bool first = true;
        for (const BatteryBoard& board : kBoards)
            if (wanted(board, cfg.batteryHat)) {
                std::cerr << (first ? " " : ", ") << hex8(board.inaAddr)
                          << " for the " << board.name;
                first = false;
            }
        std::cerr << "); running without a gauge\n";
        return nullptr;
    }

    b->desc_ = std::string("Waveshare ") + b->board_->name + " @ " + path +
               " " + hex8(b->board_->inaAddr);
    b->sample(); // seed the smoothed voltage so the first frame shows a level
    b->lastPoll_ = Clock::now();
    return b;
}

bool Battery::sample() {
    if (!board_) return false; // nothing adopted yet
    const BatteryBoard& board = *board_;

    // The INA219 loses its calibration on a bus glitch or a brown-out, which
    // silently zeroes the current and power registers. Waveshare's drivers
    // rewrite it before every read; do the same rather than trust it to stick.
    if (!writeReg(board.inaAddr, kRegCal, board.calibration)) return false;

    std::uint16_t bus = 0, shunt = 0, current = 0, power = 0;
    if (!readReg(board.inaAddr, kRegBus, bus) ||
        !readReg(board.inaAddr, kRegShunt, shunt) ||
        !readReg(board.inaAddr, kRegCurrent, current) ||
        !readReg(board.inaAddr, kRegPower, power))
        return false;

    (void)shunt; // read to keep the sequence identical to the reference driver

    // Bus register: bits 15..3 hold the voltage, the low bits are status flags.
    double volts = (double)(bus >> 3) * kBusLsbV;
    if (volts < board.presentV) {
        st_.valid = false; // no pack, or the ADC hasn't produced a reading yet
        return false;
    }

    // Normalise to "+ = into the pack" regardless of how the board's shunt is
    // wired; on the (D) that means the negation its demo applies.
    double amps = (double)signed16(current) * board.currentLsbA;
    if (board.dischargePositive) amps = -amps;
    double watts = (double)signed16(power) * board.powerLsbW;

    smoothedV_ = (smoothedV_ <= 0.0)
                     ? volts
                     : smoothedV_ + kSmoothing * (volts - smoothedV_);

    double pct = (smoothedV_ - board.emptyV) / (board.fullV - board.emptyV) * 100.0;

    st_.valid = true;
    st_.volts = volts;
    st_.amps = amps;
    st_.watts = watts;
    st_.percent = (int)std::lround(std::clamp(pct, 0.0, 100.0));
    st_.charging = amps > kChargingA;
    return true;
}

bool Battery::armAutoRestart() {
    // The (B) has no MCU to ask: it comes back only when its button is pressed.
    if (!board_ || board_->mcuAddr == 0) return false;

    // Register 0x01 of the HAT's MCU: 0x55 makes it power the Pi back up by
    // itself once the pack has recovered, instead of staying off until someone
    // presses the button. It is a single-byte write, so not writeReg().
    std::uint8_t buf[2] = {0x01, 0x55};
    i2c_msg msg{};
    msg.addr = board_->mcuAddr;
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
bool Battery::adopt(const BatteryBoard&) { return false; }
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

    // Track how long the pack has been under the cut-off while off charge. A
    // single dip during a capture burst is normal, so only a sustained low
    // reading counts as critical.
    // A successful sample() guarantees a board was adopted.
    const double cutoff = board_->cutoffV;
    const bool under = st_.volts < cutoff && !st_.charging;
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
        std::cerr << "battery: " << st_.volts << " V is below the " << cutoff
                  << " V cut-off; powering off in " << left
                  << " s unless charged\n";
    }
}

} // namespace olc
