#pragma once

#include "RTU.hpp"

namespace irt {

class PAModelStat
{
 public:
  PAModelStat() = default;
  ~PAModelStat() = default;
  // getter
  irt_int get_total_pin_num() { return _total_pin_num; }
  irt_int get_track_grid_pin_num() { return _track_grid_pin_num; }
  irt_int get_track_center_pin_num() { return _track_center_pin_num; }
  irt_int get_shape_center_pin_num() { return _shape_center_pin_num; }
  irt_int get_total_port_num() { return _total_port_num; }
  std::map<irt_int, irt_int>& get_layer_port_num_map() { return _layer_port_num_map; }
  // setter
  // function
  void addTotalPinNum(const irt_int pin_num) { _total_pin_num += pin_num; }
  void addTrackGridPinNum(const irt_int pin_num) { _track_grid_pin_num += pin_num; }
  void addTrackCenterPinNum(const irt_int pin_num) { _track_center_pin_num += pin_num; }
  void addShapeCenterPinNum(const irt_int pin_num) { _shape_center_pin_num += pin_num; }
  void addTotalPortNum(const irt_int port_num) { _total_port_num += port_num; }

 private:
  irt_int _total_pin_num = 0;
  irt_int _track_grid_pin_num = 0;
  irt_int _track_center_pin_num = 0;
  irt_int _shape_center_pin_num = 0;
  irt_int _total_port_num = 0;
  std::map<irt_int, irt_int> _layer_port_num_map;
};

}  // namespace irt
