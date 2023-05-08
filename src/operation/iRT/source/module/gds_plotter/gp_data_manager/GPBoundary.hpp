#pragma once

#include "GPDataType.hpp"
#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "RTU.hpp"

namespace irt {

class GPBoundary : public LayerRect
{
 public:
  GPBoundary() = default;
  GPBoundary(const LayerRect& rect, const irt_int data_type) : LayerRect(rect) { _data_type = data_type; }
  GPBoundary(const PlanarRect& rect, const irt_int layer_idx, const irt_int data_type) : LayerRect(rect, layer_idx)
  {
    _data_type = data_type;
  }
  ~GPBoundary() = default;
  // getter
  irt_int get_data_type() const { return _data_type; }
  // setter
  void set_data_type(const irt_int data_type) { _data_type = data_type; }

  // function

 private:
  irt_int _data_type = 0;
};

}  // namespace irt
