#pragma once

#include <string>
#include <vector>

#include "EXTLayerRect.hpp"
#include "LayerCoord.hpp"
#include "Pin.hpp"

namespace irt {

class DRPin : public Pin
{
 public:
  DRPin() = default;
  explicit DRPin(const Pin& pin) : Pin(pin) {}
  ~DRPin() = default;
  // getter

  // setter

  // function

 private:
};

}  // namespace irt
