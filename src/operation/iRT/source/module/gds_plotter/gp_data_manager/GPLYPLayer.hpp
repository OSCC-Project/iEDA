#pragma once

#include <string>

#include "RTU.hpp"

namespace irt {

class GPLYPLayer
{
 public:
  GPLYPLayer(std::string color, std::string pattern, std::string layer_name, irt_int layer_idx, irt_int data_type)
  {
    _color = color;
    _pattern = pattern;
    _layer_name = layer_name;
    _layer_idx = layer_idx;
    _data_type = data_type;
  }
  ~GPLYPLayer() = default;
  // getter
  std::string& get_color() { return _color; }
  std::string& get_pattern() { return _pattern; }
  std::string& get_layer_name() { return _layer_name; }
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_data_type() const { return _data_type; }
  // setter
  void set_color(const std::string& color) { _color = color; }
  void set_pattern(const std::string& pattern) { _pattern = pattern; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_data_type(const irt_int data_type) { _data_type = data_type; }

  // function

 private:
  std::string _color;
  std::string _pattern;
  std::string _layer_name;
  irt_int _layer_idx;
  irt_int _data_type;
};

}  // namespace irt
