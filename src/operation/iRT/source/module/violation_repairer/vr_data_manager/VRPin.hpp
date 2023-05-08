#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class VRPin : public Pin
{
 public:
  VRPin() = default;
  explicit VRPin(const Pin& pin) : Pin(pin) {}
  ~VRPin() = default;

  // getter

  // setter

  // function

 private:
};

}  // namespace irt
