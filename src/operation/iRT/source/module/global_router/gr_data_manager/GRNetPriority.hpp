#pragma once

#include "RTU.hpp"

namespace irt {

class GRNetPriority
{
 public:
  GRNetPriority() = default;
  ~GRNetPriority() = default;
  // getter
  ConnectType get_connect_type() const { return _connect_type; }
  double get_routing_area() const { return _routing_area; }
  double get_length_width_ratio() const { return _length_width_ratio; }
  irt_int get_pin_num() const { return _pin_num; }

  // setter
  void set_connect_type(const ConnectType connect_type) { _connect_type = connect_type; }
  void set_routing_area(const double routing_area) { _routing_area = routing_area; }
  void set_length_width_ratio(const double length_width_ratio) { _length_width_ratio = length_width_ratio; }
  void set_pin_num(const irt_int pin_num) { _pin_num = pin_num; }

  // function

 private:
  ConnectType _connect_type = ConnectType::kNone;
  double _routing_area = -1;
  double _length_width_ratio = 1;
  irt_int _pin_num = -1;
};

}  // namespace irt
