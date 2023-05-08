#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class PAPin : public Pin
{
 public:
  PAPin() = default;
  explicit PAPin(const Pin& pin) : Pin(pin) {}
  ~PAPin() = default;
  // getter

  // setter

  // function

 private:
};

}  // namespace irt
