#pragma once

#include "DRGroup.hpp"
#include "DRTaskPriority.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "RTNode.hpp"

namespace irt {

class DRTask
{
 public:
  DRTask() = default;
  ~DRTask() = default;
  // getter
  irt_int get_origin_net_idx() { return _origin_net_idx; }
  TNode<RTNode>* get_origin_node() { return _origin_node; }
  irt_int get_task_idx() { return _task_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<DRGroup>& get_dr_group_list() { return _dr_group_list; }
  std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_coord_cost_map() { return _coord_cost_map; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  DRTaskPriority& get_dr_task_priority() { return _dr_task_priority; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  // setter
  void set_origin_net_idx(const irt_int origin_net_idx) { _origin_net_idx = origin_net_idx; }
  void set_origin_node(TNode<RTNode>* origin_node) { _origin_node = origin_node; }
  void set_task_idx(const irt_int task_idx) { _task_idx = task_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; };
  void set_dr_group_list(const std::vector<DRGroup>& dr_group_list) { _dr_group_list = dr_group_list; }
  void set_coord_cost_map(const std::map<LayerCoord, double, CmpLayerCoordByXASC>& coord_cost_map) { _coord_cost_map = coord_cost_map; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_dr_task_priority(const DRTaskPriority& dr_task_priority) { _dr_task_priority = dr_task_priority; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // function

 private:
  irt_int _origin_net_idx = -1;
  TNode<RTNode>* _origin_node = nullptr;
  irt_int _task_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<DRGroup> _dr_group_list;
  std::map<LayerCoord, double, CmpLayerCoordByXASC> _coord_cost_map;
  PlanarRect _bounding_box;
  DRTaskPriority _dr_task_priority;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
};

}  // namespace irt
