/**
 * @file SdcSetIODelay.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of set_input_delay and set_output_delay constrain.
 * @version 0.1
 * @date 2021-05-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "SdcSetIODelay.hh"

namespace ista {

SdcSetIODelay::SdcSetIODelay(const char* constrain_name, const char* clock_name,
                             double delay_value)
    : SdcIOConstrain(constrain_name),
      _rise(1),
      _fall(1),
      _max(1),
      _min(1),
      _clock_fall(0),
      _reserved(0),
      _clock_name(clock_name),
      _delay_value(delay_value) {}

SdcSetInputDelay::SdcSetInputDelay(const char* constrain_name,
                                   const char* clock_name, double delay_value)
    : SdcSetIODelay(constrain_name, clock_name, delay_value) {}

SdcSetOutputDelay::SdcSetOutputDelay(const char* constrain_name,
                                     const char* clock_name, double delay_value)
    : SdcSetIODelay(constrain_name, clock_name, delay_value) {}

}  // namespace ista