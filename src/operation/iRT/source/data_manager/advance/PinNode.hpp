#pragma once

#include "EXTLayerCoord.hpp"

namespace irt {

class PinNode : public EXTLayerCoord
{
 public:
  PinNode() = default;
  PinNode(const PinNode& other) : EXTLayerCoord(other)
  {
    _net_idx = other._net_idx;
    _pin_idx = other._pin_idx;
  }
  ~PinNode() = default;

  // getter
  irt_int get_net_idx() const { return _net_idx; }
  irt_int get_pin_idx() const { return _pin_idx; }

  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_pin_idx(const irt_int pin_idx) { _pin_idx = pin_idx; }
  // function

 private:
  irt_int _net_idx = -1;
  irt_int _pin_idx = -1;
};

}  // namespace irt
