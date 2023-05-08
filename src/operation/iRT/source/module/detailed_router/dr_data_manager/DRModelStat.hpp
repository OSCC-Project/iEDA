#pragma once

#include "RTU.hpp"

namespace irt {

class DRModelStat
{
 public:
  DRModelStat() = default;
  ~DRModelStat() = default;
  // getter
  double get_total_wire_length() { return _total_wire_length; }
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  irt_int get_total_via_number() { return _total_via_number; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  double get_total_net_and_net_violation_area() { return _total_net_and_net_violation_area; }
  std::map<irt_int, double>& get_routing_net_and_net_violation_area_map() { return _routing_net_and_net_violation_area_map; }
  double get_total_net_and_obs_violation_area() { return _total_net_and_obs_violation_area; }
  std::map<irt_int, double>& get_routing_net_and_obs_violation_area_map() { return _routing_net_and_obs_violation_area_map; }
  // setter
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addTotalViaNumber(const irt_int via_number) { _total_via_number += via_number; }
  void addTotalNetAndNetViolation(const double violation_area) { _total_net_and_net_violation_area += violation_area; }
  void addTotalNetAndObsViolation(const double violation_area) { _total_net_and_obs_violation_area += violation_area; }

 private:
  double _total_wire_length = 0;
  std::map<irt_int, double> _routing_wire_length_map;
  irt_int _total_via_number = 0;
  std::map<irt_int, irt_int> _cut_via_number_map;
  double _total_net_and_net_violation_area = 0;
  std::map<irt_int, double> _routing_net_and_net_violation_area_map;
  double _total_net_and_obs_violation_area = 0;
  std::map<irt_int, double> _routing_net_and_obs_violation_area_map;
};

}  // namespace irt
