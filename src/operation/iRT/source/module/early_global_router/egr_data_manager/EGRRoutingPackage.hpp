// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include "LayerCoord.hpp"
#include "Segment.hpp"
#include "flute3/flute.h"

namespace irt {

class EGRRoutingPackage
{
 public:
  EGRRoutingPackage() = default;
  ~EGRRoutingPackage() = default;

  // getter
  std::vector<LayerCoord>& get_pin_coord_list() { return _pin_coord_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  LayerCoord& get_pin_coord() { return _pin_coord; }
  LayerCoord& get_seg_coord() { return _seg_coord; }
  irt_int get_number_calculated() { return _number_of_segments_with_distance_calculated; }
  std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC>& get_min_distance_map() { return _min_distance_map; }
  std::vector<std::pair<LayerCoord, LayerCoord>>& get_topo_coord_pair_list() { return _topo_coord_pair_list; }
  std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC>& get_planar_coord_layer_map() { return _planar_coord_layer_map; }
  Flute::Tree& get_flute_tree() { return _flute_tree; }

  // setter
  void set_pin_coord_list(const std::vector<LayerCoord>& pin_coord_list) { _pin_coord_list = pin_coord_list; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  void set_pin_coord(const LayerCoord& pin_coord) { _pin_coord = pin_coord; }
  void set_seg_coord(const LayerCoord& seg_coord) { _seg_coord = seg_coord; }
  void set_number_calculated(irt_int number) { _number_of_segments_with_distance_calculated = number; }
  void set_min_distance_map(std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC>& min_distance_map)
  {
    _min_distance_map = min_distance_map;
  }
  void set_topo_coord_pair_list(const std::vector<std::pair<LayerCoord, LayerCoord>>& topo_coord_pair_list)
  {
    _topo_coord_pair_list = topo_coord_pair_list;
  }
  void set_planar_coord_layer_map(const std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC>& planar_coord_layer_map)
  {
    _planar_coord_layer_map = planar_coord_layer_map;
  }
  void set_flute_tree(const Flute::Tree& flute_tree) { _flute_tree = flute_tree; }

  // function
  bool continueRouting() { return !_pin_coord_list.empty(); }

 private:
  std::vector<LayerCoord> _pin_coord_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;

  // gradual router
  irt_int _number_of_segments_with_distance_calculated;
  std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC> _min_distance_map;
  LayerCoord _pin_coord;
  LayerCoord _seg_coord;

  // topo router
  std::vector<std::pair<LayerCoord, LayerCoord>> _topo_coord_pair_list;
  std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC> _planar_coord_layer_map;
  Flute::Tree _flute_tree;
};

}  // namespace irt
