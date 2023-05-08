#pragma once

#include "EXTLayerRect.hpp"
#include "RTU.hpp"
#include "RoutingLayer.hpp"
#include "ViaMaster.hpp"

namespace irt {

class Blockage : public EXTLayerRect
{
 public:
  Blockage() = default;
  ~Blockage() = default;
  // getter
  // setter
  void set_is_artificial(const bool is_artificial) { _is_artificial = is_artificial; }
  // function
  bool isArtificial() const { return _is_artificial; }

 private:
  bool _is_artificial = false;
};

}  // namespace irt
