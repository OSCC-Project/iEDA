#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class RAPin : public Pin
{
 public:
  RAPin() = default;
  explicit RAPin(const Pin& pin) : Pin(pin) {}
  ~RAPin() = default;
  // getter

  // setter

  // function

 private:
};

}  // namespace irt
