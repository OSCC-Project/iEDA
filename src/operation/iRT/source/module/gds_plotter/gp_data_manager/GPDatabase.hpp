#pragma once

#include "EXTPlanarRect.hpp"
#include "RoutingLayer.hpp"
#include "ViaMaster.hpp"

namespace irt {

class GPDatabase
{
 public:
  GPDatabase() = default;
  ~GPDatabase() = default;
  // getter
  GCellAxis& get_gcell_axis() { return _gcell_axis; }
  EXTPlanarRect& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  std::vector<Blockage>& get_cut_blockage_list() { return _cut_blockage_list; }
  std::map<irt_int, irt_int>& get_routing_layer_gds_map() { return _routing_layer_gds_map; }
  std::map<irt_int, irt_int>& get_cut_layer_gds_map() { return _cut_layer_gds_map; }
  std::map<irt_int, irt_int>& get_gds_routing_layer_map() { return _gds_routing_layer_map; }
  std::map<irt_int, irt_int>& get_gds_cut_layer_map() { return _gds_cut_layer_map; }
  // setter

  // function

 private:
  GCellAxis _gcell_axis;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
  std::vector<Blockage> _cut_blockage_list;
  std::map<irt_int, irt_int> _routing_layer_gds_map;
  std::map<irt_int, irt_int> _cut_layer_gds_map;
  std::map<irt_int, irt_int> _gds_routing_layer_map;
  std::map<irt_int, irt_int> _gds_cut_layer_map;
};

}  // namespace irt
