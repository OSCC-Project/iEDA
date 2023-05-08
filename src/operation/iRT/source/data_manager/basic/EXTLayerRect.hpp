#pragma once

#include "EXTPlanarRect.hpp"

namespace irt {

class EXTLayerRect : public EXTPlanarRect
{
 public:
  EXTLayerRect() = default;
  ~EXTLayerRect() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  irt_int _layer_idx = -1;
};

}  // namespace irt
