// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file Time.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The utility class of time.
 * @version 0.1
 * @date 2021-05-08
 */
#pragma once

#include "absl/time/civil_time.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace ieda {

/**
 * @brief The utility class relate to the time.
 *
 */
class Time {
 public:
  enum TimeMode {
    kOneShot = 0,  // start the timer
    kAccumulate,       // stop the timer
  };
  /**
   * @brief Get now wall time such as 2021-05-08T17:06:42.
   *
   * @return const char*
   */
  static const char* getNowWallTime();

  /**
   * @brief Start the timer
   */
  static void start() { _start_time = absl::Now(); }

  /**
   * @brief Stop the timer
   */
  static void stop() { 
    _end_time = absl::Now();
    if (_time_mode == kAccumulate) {
      _accumulate_time += absl::ToDoubleSeconds(_end_time - _start_time);
    }
  }

  static void set_time_mode(TimeMode mode) { _time_mode = mode; }
  static TimeMode get_time_mode() { return _time_mode; }

  static void resetAccumulateTime() { _accumulate_time = 0; }
  static double get_accumulate_time() { return _accumulate_time; }

  /**
   * @brief Get the elapsed time in seconds
   *
   * @return double The elapsed time in seconds
   */
  static double elapsedTime() { return absl::ToDoubleSeconds(_end_time - _start_time); }

 private:
  static absl::Time _start_time;
  static absl::Time _end_time;

  static TimeMode _time_mode;
  static double _accumulate_time;
};

}  // namespace ieda