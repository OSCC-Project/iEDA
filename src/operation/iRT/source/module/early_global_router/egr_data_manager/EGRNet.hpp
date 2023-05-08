#pragma once

#include "EGRPin.hpp"
#include "Segment.hpp"

namespace irt {

class EGRNet
{
 public:
  EGRNet() = default;
  ~EGRNet() = default;
  // getter
  std::string get_net_name() { return _net_name; }
  std::vector<EGRPin>& get_pin_list() { return _pin_list; }
  EGRPin& get_driving_pin() { return _driving_pin; }
  MTree<LayerCoord>& get_coord_tree() { return _coord_tree; }
  // setter
  void set_net_name(std::string net_name) { _net_name = net_name; }
  void set_pin_list(const std::vector<EGRPin>& pin_list) { _pin_list = pin_list; }
  void set_driving_pin(const EGRPin& driving_pin) { _driving_pin = driving_pin; }
  void set_coord_tree(const MTree<LayerCoord>& coord_tree) { _coord_tree = coord_tree; }

 private:
  std::string _net_name;
  std::vector<EGRPin> _pin_list;
  EGRPin _driving_pin;
  MTree<LayerCoord> _coord_tree;
};

}  // namespace irt
