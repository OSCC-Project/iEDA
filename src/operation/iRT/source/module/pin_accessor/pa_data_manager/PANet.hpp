#pragma once

#include "Net.hpp"
#include "PAPin.hpp"

namespace irt {

class PANet
{
 public:
  PANet() = default;
  ~PANet() = default;
  // getter
  Net* get_origin_net() { return _origin_net; }
  irt_int get_net_idx() const { return _net_idx; }
  std::string& get_net_name() { return _net_name; }
  std::vector<PAPin>& get_pa_pin_list() { return _pa_pin_list; }
  PAPin& get_pa_driving_pin() { return _pa_driving_pin; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  LayerCoord& get_balance_coord() { return _balance_coord; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_pa_pin_list(const std::vector<PAPin>& pa_pin_list) { _pa_pin_list = pa_pin_list; }
  void set_pa_driving_pin(const PAPin& pa_driving_pin) { _pa_driving_pin = pa_driving_pin; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  void set_balance_coord(const LayerCoord& balance_coord) { _balance_coord = balance_coord; }
  // function

 private:
  Net* _origin_net = nullptr;
  irt_int _net_idx = -1;
  std::string _net_name;
  std::vector<PAPin> _pa_pin_list;
  PAPin _pa_driving_pin;
  BoundingBox _bounding_box;
  LayerCoord _balance_coord;
};

}  // namespace irt
