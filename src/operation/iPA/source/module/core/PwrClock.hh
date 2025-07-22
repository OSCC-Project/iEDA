// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
};
}  // namespace ipower
