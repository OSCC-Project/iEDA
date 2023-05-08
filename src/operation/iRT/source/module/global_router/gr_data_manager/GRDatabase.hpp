#pragma once

#include "EXTPlanarRect.hpp"
#include "GRNet.hpp"
#include "RoutingLayer.hpp"

namespace irt {

class GRDatabase
{
 public:
  GRDatabase() = default;
  ~GRDatabase() = default;
  // getter
  irt_int get_micron_dbu() const { return _micron_dbu; }
  GCellAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  // setter
  void set_micron_dbu(const irt_int micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  irt_int _micron_dbu = -1;
  GCellAxis _gcell_axis;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
};

}  // namespace irt
