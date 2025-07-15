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
 * @file Time.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The utility class implemention of time.
 * @version 0.1
 * @date 2021-05-08
 */

#include "Time.hh"

#include <iostream>
#include <sstream>

#include "string/Str.hh"

namespace ieda {
/**
 * @brief Get now wall time such as 2021-05-08T17:06:42.
 *
 * @return const char*
 */
const char* Time::getNowWallTime()
{
  absl::Time t = absl::Now();
  absl::TimeZone loc = absl::LocalTimeZone();
  const auto now_time = loc.At(t);
  std::ostringstream s;
  s << now_time.cs;

  return Str::printf("%s", s.str().c_str());
}

// Initialize static members
absl::Time Time::_start_time = absl::UnixEpoch();
absl::Time Time::_end_time = absl::UnixEpoch();
Time::TimeMode Time::_time_mode = Time::kOneShot;
double Time::_accumulate_time = 0.0;

}  // namespace ieda
