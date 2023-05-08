#pragma once

#include "PlanarCoord.hpp"

namespace irt {

enum class PACheckType
{
  kNone = 0,
  kPrefOrien = 1,
  kViaOrien = 2
};

class PANode : public AccessPoint
{
 public:
  PANode() = default;
  explicit PANode(const AccessPoint& access_point) : AccessPoint(access_point) {}
  ~PANode() = default;
  // getter
  irt_int get_net_idx() { return _net_idx; }
  PAPin* get_pin_ptr() { return _pin_ptr; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_pin_ptr(PAPin* pin_ptr) { _pin_ptr = pin_ptr; }

 private:
  irt_int _net_idx = -1;
  PAPin* _pin_ptr = nullptr;
};

}  // namespace irt
