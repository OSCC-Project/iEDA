
#include "SdcTimingDRC.hh"

namespace ista {

SetMaxTransition::SetMaxTransition(double transition_value)
    : SdcTimingDRC(transition_value),
      _is_clock_path(0),
      _is_data_path(0),
      _is_rise(0),
      _is_fall(0) {}

SetMaxCapacitance::SetMaxCapacitance(double capacitance_value)
    : SdcTimingDRC(capacitance_value),
      _is_clock_path(0),
      _is_data_path(0),
      _is_rise(0),
      _is_fall(0) {}

SetMaxFanout::SetMaxFanout(double fanout_value) : SdcTimingDRC(fanout_value) {}

}  // namespace ista
