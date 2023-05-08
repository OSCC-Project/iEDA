#pragma once

#include "RTU.hpp"

namespace irt {

class CutLayer
{
 public:
  CutLayer() = default;
  ~CutLayer() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const irt_int layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }

  // function

 private:
  irt_int _layer_idx = -1;
  irt_int _layer_order = -1;
  std::string _layer_name;
};

}  // namespace irt
