#pragma once

#include "GCellAxis.hpp"
#include "RoutingLayer.hpp"

namespace irt {

class PADatabase
{
 public:
  PADatabase() = default;
  ~PADatabase() = default;
  // getter
  GCellAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  // setter
  // function

 private:
  GCellAxis _gcell_axis;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
};

}  // namespace irt
