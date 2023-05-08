#pragma once

#include "LayerRect.hpp"

namespace irt {

class ViaMaster
{
 public:
  ViaMaster() = default;
  ~ViaMaster() = default;
  // getter
  std::pair<irt_int, irt_int>& get_via_idx() { return _via_idx; }
  std::string& get_via_name() { return _via_name; }
  LayerRect& get_above_enclosure() { return _above_enclosure; }
  Direction get_above_direction() const { return _above_direction; }
  LayerRect& get_below_enclosure() { return _below_enclosure; }
  Direction get_below_direction() const { return _below_direction; }
  std::vector<PlanarRect>& get_cut_shape_list() { return _cut_shape_list; }
  irt_int get_cut_layer_idx() const { return _cut_layer_idx; }
  // setter
  void set_via_idx(const std::pair<irt_int, irt_int>& via_idx) { _via_idx = via_idx; }
  void set_via_name(const std::string& via_name) { _via_name = via_name; }
  void set_above_enclosure(const LayerRect& above_enclosure) { _above_enclosure = above_enclosure; }
  void set_above_direction(const Direction& above_direction) { _above_direction = above_direction; }
  void set_below_enclosure(const LayerRect& below_enclosure) { _below_enclosure = below_enclosure; }
  void set_below_direction(const Direction& below_direction) { _below_direction = below_direction; }
  void set_cut_shape_list(const std::vector<PlanarRect>& cut_shape_list) { _cut_shape_list = cut_shape_list; }
  void set_cut_layer_idx(const irt_int cut_layer_idx) { _cut_layer_idx = cut_layer_idx; }
  // function

 private:
  std::pair<irt_int, irt_int> _via_idx;  //<! below_layer_idx, idx
  std::string _via_name;
  LayerRect _above_enclosure;
  Direction _above_direction = Direction::kNone;
  LayerRect _below_enclosure;
  Direction _below_direction = Direction::kNone;
  std::vector<PlanarRect> _cut_shape_list;
  irt_int _cut_layer_idx;
};

}  // namespace irt
