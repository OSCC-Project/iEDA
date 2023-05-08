#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class GRPin : public Pin
{
 public:
  GRPin() = default;
  explicit GRPin(const Pin& pin) : Pin(pin) {}
  ~GRPin() = default;
  // getter

  // setter

  // function

 private:
};

}  // namespace irt
