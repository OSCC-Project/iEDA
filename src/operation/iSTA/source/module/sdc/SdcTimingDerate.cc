/**
 * @file SdcTimingDerate.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-09-25
 */

#include "SdcTimingDerate.hh"

namespace ista {

SdcTimingDerate::SdcTimingDerate(double derate_value)
    : _is_cell_delay(0),
      _is_net_delay(0),
      _is_clock_delay(0),
      _is_data_delay(0),
      _is_early_delay(0),
      _is_late_delay(0),
      _derate_value(derate_value) {}

}  // namespace ista
