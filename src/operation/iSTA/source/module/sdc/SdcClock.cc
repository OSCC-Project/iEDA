/**
 * @file sdcClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the clock command obj of sdc.
 * @version 0.1
 * @date 2020-11-22
 *
 */

#include "SdcClock.hh"

namespace ista {
SdcClock::SdcClock(const char* clock_name)
    : _clock_name(clock_name), _period(0.0) {}

SdcGenerateCLock::SdcGenerateCLock(const char* clock_name)
    : SdcClock(clock_name) {}

};  // namespace ista
