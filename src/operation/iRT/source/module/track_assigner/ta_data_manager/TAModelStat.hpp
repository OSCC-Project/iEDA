#pragma once

#include "RTU.hpp"

namespace irt {

class TAModelStat
{
 public:
  TAModelStat() = default;
  ~TAModelStat() = default;
  // getter
  double get_total_wire_length() { return _total_wire_length; }
  std::map<irt_int, double>& get_layer_wire_length_map() { return _layer_wire_length_map; }
  double get_total_net_and_net_violation_area() { return _total_net_and_net_violation_area; }
  std::map<irt_int, double>& get_layer_net_and_net_violation_area_map() { return _layer_net_and_net_violation_area_map; }
  double get_total_net_and_obs_violation_area() { return _total_net_and_obs_violation_area; }
  std::map<irt_int, double>& get_layer_net_and_obs_violation_area_map() { return _layer_net_and_obs_violation_area_map; }
  // setter
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addTotalNetAndNetViolation(const double violation_area) { _total_net_and_net_violation_area += violation_area; }
  void addTotalNetAndObsViolation(const double violation_area) { _total_net_and_obs_violation_area += violation_area; }

 private:
  double _total_wire_length = 0;
  std::map<irt_int, double> _layer_wire_length_map;
  double _total_net_and_net_violation_area = 0;
  std::map<irt_int, double> _layer_net_and_net_violation_area_map;
  double _total_net_and_obs_violation_area = 0;
  std::map<irt_int, double> _layer_net_and_obs_violation_area_map;
};

}  // namespace irt
