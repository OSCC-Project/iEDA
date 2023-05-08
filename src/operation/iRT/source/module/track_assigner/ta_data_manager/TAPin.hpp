#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class TAPin : public Pin
{
 public:
  TAPin() = default;
  explicit TAPin(const Pin& pin) : Pin(pin) {}
  ~TAPin() = default;

  // getter

  // setter

  // function

 private:
};

}  // namespace irt
