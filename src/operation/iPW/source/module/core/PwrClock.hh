/**
 * @file PwrClock.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power clock class, used for data toggle calculation relative to
 * clock.
 * @version 0.1
 * @date 2023-04-08
 */
#pragma once

#include <string>

#include "DisallowCopyAssign.hh"

namespace ipower {

/**
 * @brief The power clock, which choose the fastest clock.
 *
 */
class PwrClock {
 public:
  explicit PwrClock() = default;
  PwrClock(std::string clock_name, double clock_period_ns)
      : _clock_name(std::move(clock_name)), _clock_period_ns(clock_period_ns) {}
  ~PwrClock() = default;
  PwrClock(PwrClock&&) = default;
  PwrClock& operator=(PwrClock&&) = default;

  [[nodiscard]] auto& get_clock_name() { return _clock_name; }
  void set_clock_name(const char* clock_name) { _clock_name = clock_name; }

  [[nodiscard]] double get_clock_period_ns() const { return _clock_period_ns; }
  void set_clock_period_ns(double clock_period_ns) {
    _clock_period_ns = clock_period_ns;
  }

 private:
  std::string _clock_name;
  double _clock_period_ns =
      10.0;  //!< The default fastest clock period is 10 ns.

  DISALLOW_COPY_AND_ASSIGN(PwrClock);
};
}  // namespace ipower
