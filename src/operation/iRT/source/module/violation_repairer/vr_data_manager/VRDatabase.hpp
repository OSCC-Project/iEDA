#pragma once

namespace irt {

class VRDatabase
{
 public:
  VRDatabase() = default;
  ~VRDatabase() = default;
  // getter
  irt_int get_micron_dbu() const { return _micron_dbu; }
  GCellAxis& get_gcell_axis() { return _gcell_axis; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  // setter
  void set_micron_dbu(const irt_int micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  irt_int _micron_dbu = -1;
  GCellAxis _gcell_axis;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
};

}  // namespace irt
