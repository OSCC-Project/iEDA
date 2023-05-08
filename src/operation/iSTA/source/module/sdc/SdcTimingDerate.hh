/**
 * @file SdcTimingDerate.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Set timing derate for ocv analysis.
 * @version 0.1
 * @date 2021-09-24
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include "SdcCommand.hh"

namespace ista {

/**
 * @brief The sdc set_timing_derate cmd information.
 *
 */
class SdcTimingDerate : public SdcCommandObj {
 public:
  explicit SdcTimingDerate(double derate_value);
  ~SdcTimingDerate() override = default;

  unsigned isCellDelay() const { return _is_cell_delay; }
  void set_is_cell_delay(bool is_set) { _is_cell_delay = is_set ? 1 : 0; }

  unsigned isNetDelay() const { return _is_net_delay; }
  void set_is_net_delay(bool is_set) { _is_net_delay = is_set ? 1 : 0; }

  unsigned isClockDelay() const { return _is_clock_delay; }
  void set_is_clock_delay(bool is_set) { _is_clock_delay = is_set ? 1 : 0; }

  unsigned isDataDelay() const { return _is_data_delay; }
  void set_is_data_delay(bool is_set) { _is_data_delay = is_set ? 1 : 0; }

  unsigned isEarlyDelay() const { return _is_early_delay; }
  void set_is_early_delay(bool is_set) { _is_early_delay = is_set ? 1 : 0; }

  unsigned isLateDelay() const { return _is_late_delay; }
  void set_is_late_delay(bool is_set) { _is_late_delay = is_set ? 1 : 0; }

  double get_derate_value() const { return _derate_value; }

 private:
  unsigned _is_cell_delay : 1;
  unsigned _is_net_delay : 1;
  unsigned _is_clock_delay : 1;
  unsigned _is_data_delay : 1;
  unsigned _is_early_delay : 1;
  unsigned _is_late_delay : 1;
  unsigned _reserved : 26;

  double _derate_value;
};

}  // namespace ista
