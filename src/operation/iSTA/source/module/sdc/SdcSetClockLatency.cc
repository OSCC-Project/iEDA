/**
 * @file SdcSetClockLatency.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_uncertainty implemention.
 * @version 0.1
 * @date 2021-10-21
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "SdcSetClockLatency.hh"

namespace ista {

SdcSetClockLatency::SdcSetClockLatency(double delay_value)
    : _rise(0),
      _fall(0),
      _max(0),
      _min(0),
      _early(0),
      _late(0),
      _reserved(0),
      _delay_value(delay_value) {}

}  // namespace ista
