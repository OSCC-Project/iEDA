/**
 * @file SdcSetClockUncertainty.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_uncertainty implemention.
 * @version 0.1
 * @date 2021-10-25
 */

#include "SdcSetClockUncertainty.hh"

namespace ista {

SdcSetClockUncertainty::SdcSetClockUncertainty(double uncertainty_value)
    : _rise(1),
      _fall(1),
      _setup(1),
      _hold(1),
      _reserved(0),
      _uncertainty_value(uncertainty_value) {}

}  // namespace ista
