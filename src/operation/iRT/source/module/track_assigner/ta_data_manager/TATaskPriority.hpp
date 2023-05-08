#pragma once

#include "RTU.hpp"

namespace irt {

class TATaskPriority
{
 public:
  TATaskPriority() = default;
  ~TATaskPriority() = default;
  // getter
  ConnectType get_connect_type() const { return _connect_type; }
  double get_length_width_ratio() const { return _length_width_ratio; }

  // setter
  void set_connect_type(const ConnectType connect_type) { _connect_type = connect_type; }
  void set_length_width_ratio(const double length_width_ratio) { _length_width_ratio = length_width_ratio; }

  // function

 private:
  ConnectType _connect_type = ConnectType::kNone;
  double _length_width_ratio = 1;
};

}  // namespace irt
