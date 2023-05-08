#pragma once

#include "RTU.hpp"

namespace irt {

class TAPanelStat
{
 public:
  TAPanelStat() = default;
  ~TAPanelStat() = default;
  // getter
  double get_total_wire_length() { return _total_wire_length; }
  double get_net_and_net_violation_area() { return _net_and_net_violation_area; }
  double get_net_and_obs_violation_area() { return _net_and_obs_violation_area; }
  // setter
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addNetAndNetViolation(const double violation_area) { _net_and_net_violation_area += violation_area; }
  void addNetAndObsViolation(const double violation_area) { _net_and_obs_violation_area += violation_area; }

 private:
  double _total_wire_length = 0;
  double _net_and_net_violation_area = 0;
  double _net_and_obs_violation_area = 0;
};

}  // namespace irt
