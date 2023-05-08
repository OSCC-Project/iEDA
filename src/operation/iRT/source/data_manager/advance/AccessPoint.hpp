#pragma once

#include "AccessPointType.hpp"
#include "EXTLayerCoord.hpp"
#include "RTU.hpp"

namespace irt {

class AccessPoint : public EXTLayerCoord
{
 public:
  AccessPoint() = default;
  AccessPoint(irt_int real_x, irt_int real_y, irt_int layer_idx, AccessPointType type)
  {
    set_real_coord(real_x, real_y);
    set_layer_idx(layer_idx);
    _type = type;
  }
  ~AccessPoint() = default;
  // getter
  AccessPointType get_type() const { return _type; }
  // setter
  void set_type(const AccessPointType& type) { _type = type; }
  // function

 private:
  AccessPointType _type = AccessPointType::kNone;
};

}  // namespace irt
