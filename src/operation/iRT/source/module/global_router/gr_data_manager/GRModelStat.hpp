#pragma once

#include "RTU.hpp"

namespace irt {

class GRModelStat
{
 public:
  GRModelStat() = default;
  ~GRModelStat() = default;
  // getter
  double get_total_wire_length() const { return _total_wire_length; }
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  irt_int get_total_via_number() const { return _total_via_number; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  double get_max_wire_overflow() { return _max_wire_overflow; }
  std::vector<double>& get_wire_overflow_list() { return _wire_overflow_list; }
  double get_max_via_overflow() { return _max_via_overflow; }
  std::vector<double>& get_via_overflow_list() { return _via_overflow_list; }
  // setter
  void set_max_wire_overflow(const double max_wire_overflow) { _max_wire_overflow = max_wire_overflow; }
  void set_max_via_overflow(const double max_via_overflow) { _max_via_overflow = max_via_overflow; }
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addTotalViaNumber(const irt_int via_number) { _total_via_number += via_number; }

 private:
  double _total_wire_length = 0;
  std::map<irt_int, double> _routing_wire_length_map;
  irt_int _total_via_number = 0;
  std::map<irt_int, irt_int> _cut_via_number_map;
  double _max_wire_overflow;
  std::vector<double> _wire_overflow_list;
  double _max_via_overflow;
  std::vector<double> _via_overflow_list;
};

}  // namespace irt
