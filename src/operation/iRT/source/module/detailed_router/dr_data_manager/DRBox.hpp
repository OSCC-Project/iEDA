#pragma once

#include "DRBoxStat.hpp"
#include "DRNode.hpp"
#include "DRNodeGraph.hpp"
#include "DRTask.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"

namespace irt {

class DRBox
{
 public:
  DRBox() = default;
  ~DRBox() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid_coord; }
  std::map<irt_int, std::vector<LayerRect>>& get_net_blockage_map() { return _net_blockage_map; }
  std::map<irt_int, std::vector<LayerRect>>& get_net_region_map() { return _net_region_map; }
  std::vector<DRTask>& get_dr_task_list() { return _dr_task_list; }
  std::vector<DRNodeGraph>& get_layer_graph_list() { return _layer_graph_list; }
  DRBoxStat& get_dr_box_stat() { return _dr_box_stat; }
  // setter
  void set_grid_coord(const PlanarCoord& grid_coord) { _grid_coord = grid_coord; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<LayerRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  void set_net_region_map(const std::map<irt_int, std::vector<LayerRect>>& net_region_map) { _net_region_map = net_region_map; }
  void set_dr_task_list(const std::vector<DRTask>& dr_task_list) { _dr_task_list = dr_task_list; }
  void set_layer_graph_list(const std::vector<DRNodeGraph>& layer_graph_list) { _layer_graph_list = layer_graph_list; }
  // function
  bool skipRouting() { return _dr_task_list.empty(); }
  void freeNodeGraph()
  {
    for (DRNodeGraph& node_graph : _layer_graph_list) {
      node_graph.free();
    }
  }
  DRNode* getNodeRef(const LayerCoord& coord)
  {
    if (coord.get_layer_idx() < 0 || coord.get_layer_idx() >= static_cast<irt_int>(_layer_graph_list.size())) {
      LOG_INST.error(Loc::current(), "The coord layer_idx is error!");
    }
    DRNodeGraph& node_graph = _layer_graph_list[coord.get_layer_idx()];
    std::unordered_map<irt_int, std::unordered_map<int, int>>& x_y_idx_map = node_graph.get_x_y_idx_map();
    std::vector<DRNode>& dr_node_list = node_graph.get_dr_node_list();

    if (!RTUtil::exist(x_y_idx_map, coord.get_x())) {
      return nullptr;
    }
    if (!RTUtil::exist(x_y_idx_map[coord.get_x()], coord.get_y())) {
      return nullptr;
    }
    return &dr_node_list[x_y_idx_map[coord.get_x()][coord.get_y()]];
  }
#if 1  // astar
  double get_wire_unit() const { return _wire_unit; }
  double get_via_unit() const { return _via_unit; }
  const irt_int get_curr_task_idx() const { return _dr_task_ref->get_task_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _dr_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _dr_task_ref->get_coord_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<std::vector<DRNode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<DRNode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::vector<DRNode*>& get_path_node_comb() { return _path_node_comb; }
  std::vector<Segment<DRNode*>>& get_node_segment_list() { return _node_segment_list; }
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<DRNode*>& get_visited_node_list() { return _visited_node_list; }
  DRNode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_dr_task_ref(DRTask* dr_task_ref) { _dr_task_ref = dr_task_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<DRNode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<DRNode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_path_node_comb(const std::vector<DRNode*>& path_node_comb) { _path_node_comb = path_node_comb; }
  void set_node_segment_list(const std::vector<Segment<DRNode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  void set_forced_routing(const bool forced_routing) { _forced_routing = forced_routing; }
  void set_open_queue(const std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<DRNode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(DRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
  bool isForcedRouting() { return _forced_routing; }
#endif

 private:
  PlanarCoord _grid_coord;
  std::map<irt_int, std::vector<LayerRect>> _net_blockage_map;
  std::map<irt_int, std::vector<LayerRect>> _net_region_map;
  std::vector<DRTask> _dr_task_list;
  std::vector<DRNodeGraph> _layer_graph_list;
  DRBoxStat _dr_box_stat;
#if 1  // astar
  // config
  double _wire_unit = 1;
  double _via_unit = 2;
  // single net
  DRTask* _dr_task_ref = nullptr;
  PlanarRect _routing_region;
  std::vector<std::vector<DRNode*>> _start_node_comb_list;
  std::vector<std::vector<DRNode*>> _end_node_comb_list;
  std::vector<DRNode*> _path_node_comb;
  std::vector<Segment<DRNode*>> _node_segment_list;
  // single path
  bool _forced_routing = false;
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost> _open_queue;
  std::vector<DRNode*> _visited_node_list;
  DRNode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt
