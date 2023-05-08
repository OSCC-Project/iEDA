/**
 * @file SdcSetLoad.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set_load constrain implemention.
 * @version 0.1
 * @date 2021-04-14
 */
#include "SdcSetLoad.hh"

namespace ista {
SdcSetLoad::SdcSetLoad(const char* constrain_name, double load_value)
    : SdcIOConstrain(constrain_name),
      _rise(0),
      _fall(0),
      _max(0),
      _min(0),
      _pin_load(0),
      _wire_load(0),
      _subtract_pin_load(0),
      _allow_negative_load(0),
      _reserved(0),
      _load_value(load_value) {}

}  // namespace ista
