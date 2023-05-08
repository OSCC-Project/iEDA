/**
 * @file SdcSetInputTransition.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set_input_transition constrain implemention.
 * @version 0.1
 * @date 2021-04-14
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "SdcSetInputTransition.hh"

namespace ista {
SdcSetInputTransition::SdcSetInputTransition(const char* constrain_name,
                                             double transition_value)
    : SdcIOConstrain(constrain_name),
      _rise(1),
      _fall(1),
      _max(1),
      _min(1),
      _reserved(0),
      _transition_value(transition_value) {}

}  // namespace ista
