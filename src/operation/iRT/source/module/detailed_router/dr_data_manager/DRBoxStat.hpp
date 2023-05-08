#pragma once

#include "RTU.hpp"

namespace irt {

class DRBoxStat
{
 public:
  DRBoxStat() = default;
  ~DRBoxStat() = default;
  // getter
  double get_total_wire_length() { return _total_wire_length; }
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  irt_int get_total_via_number() { return _total_via_number; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  std::map<irt_int, double>& get_routing_net_and_net_violation_area_map() { return _routing_net_and_net_violation_area_map; }
  std::map<irt_int, double>& get_routing_net_and_obs_violation_area_map() { return _routing_net_and_obs_violation_area_map; }
  // setter
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addTotalViaNumber(const irt_int via_number) { _total_via_number += via_number; }

 private:
  double _total_wire_length = 0;
  std::map<irt_int, double> _routing_wire_length_map;
  irt_int _total_via_number = 0;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::map<irt_int, double> _routing_net_and_net_violation_area_map;
  std::map<irt_int, double> _routing_net_and_obs_violation_area_map;
};

}  // namespace irt
