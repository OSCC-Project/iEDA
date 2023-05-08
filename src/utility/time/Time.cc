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

#include "absl/time/civil_time.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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

}  // namespace ieda
