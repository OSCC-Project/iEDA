#pragma once

#include <string>
#include <vector>

#include "EXTLayerRect.hpp"
#include "LayerCoord.hpp"
#include "Pin.hpp"

namespace irt {

class EGRPin : public Pin
{
 public:
  EGRPin() = default;
  explicit EGRPin(const Pin& pin) : Pin(pin) {}
  ~EGRPin() = default;
  // getter

  // setter

  // function

 private:
};

}  // namespace irt
