#pragma once

#include "EXTLayerCoord.hpp"

namespace irt {

class ViaNode : public EXTPlanarCoord
{
 public:
  ViaNode() = default;
  ViaNode(const ViaNode& other) : EXTPlanarCoord(other)
  {
    _net_idx = other._net_idx;
    _via_idx = other._via_idx;
  }
  ~ViaNode() = default;
  // getter
  irt_int get_net_idx() const { return _net_idx; }
  std::pair<irt_int, irt_int>& get_via_idx() { return _via_idx; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_via_idx(const std::pair<irt_int, irt_int>& via_idx) { _via_idx = via_idx; }
  // function

 private:
  irt_int _net_idx = -1;
  std::pair<irt_int, irt_int> _via_idx;
};

}  // namespace irt
