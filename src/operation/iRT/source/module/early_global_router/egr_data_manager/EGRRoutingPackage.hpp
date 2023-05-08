#pragma once

#include "LayerCoord.hpp"
#include "Segment.hpp"

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
  int get_number_already_counted() { return _number_already_counted; }
  std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC>& get_min_distance_map() { return _min_distance_map; }
  std::vector<std::pair<LayerCoord, LayerCoord>>& get_topo_coord_pair_list() { return _topo_coord_pair_list; }
  std::vector<std::pair<Segment<LayerCoord>, Segment<LayerCoord>>>& get_topo_segment_pair_list() { return _topo_segment_pair_list; }
  // setter
  void set_pin_coord_list(const std::vector<LayerCoord>& pin_coord_list) { _pin_coord_list = pin_coord_list; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  void set_pin_coord(const LayerCoord& pin_coord) { _pin_coord = pin_coord; }
  void set_seg_coord(const LayerCoord& seg_coord) { _seg_coord = seg_coord; }
  void set_number_already_counted(irt_int number_already_counted) { _number_already_counted = number_already_counted; }
  void set_min_distance_map(std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC>& min_distance_map)
  {
    _min_distance_map = min_distance_map;
  }
  void set_topo_coord_pair_list(const std::vector<std::pair<LayerCoord, LayerCoord>>& topo_coord_pair_list)
  {
    _topo_coord_pair_list = topo_coord_pair_list;
  }
  void set_topo_segment_pair_list(const std::vector<std::pair<Segment<LayerCoord>, Segment<LayerCoord>>>& topo_segment_pair_list)
  {
    _topo_segment_pair_list = topo_segment_pair_list;
  }
  // function
  bool continueRouting() { return !_pin_coord_list.empty(); }

 private:
  std::vector<LayerCoord> _pin_coord_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;

  // gradual router
  irt_int _number_already_counted;
  std::map<LayerCoord, std::pair<irt_int, LayerCoord>, CmpLayerCoordByXASC> _min_distance_map;
  LayerCoord _pin_coord;
  LayerCoord _seg_coord;

  // topo router
  std::vector<std::pair<Segment<LayerCoord>, Segment<LayerCoord>>> _topo_segment_pair_list;
  std::vector<std::pair<LayerCoord, LayerCoord>> _topo_coord_pair_list;
};

}  // namespace irt
