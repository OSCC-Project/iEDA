#pragma once

#include "GRModelStat.hpp"
#include "GRNet.hpp"
#include "GRNode.hpp"
#include "GridMap.hpp"

namespace irt {

class GRModel
{
 public:
  GRModel() = default;
  ~GRModel() = default;
  // getter
  std::vector<GridMap<GRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::vector<GRNet>& get_gr_net_list() { return _gr_net_list; }
  GRModelStat& get_gr_model_stat() { return _gr_model_stat; }
  // setter
  void set_layer_node_map(const std::vector<GridMap<GRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_gr_net_list(const std::vector<GRNet>& gr_net_list) { _gr_net_list = gr_net_list; }
#if 1  // astar
  double get_wire_unit() const { return _wire_unit; }
  double get_via_unit() const { return _via_unit; }
  const irt_int get_curr_net_idx() const { return _gr_net_ref->get_net_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _gr_net_ref->get_bounding_box().get_grid_rect(); }
  const GridMap<double>& get_curr_cost_map() const { return _gr_net_ref->get_ra_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<std::vector<GRNode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<GRNode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::vector<GRNode*>& get_path_node_comb() { return _path_node_comb; }
  std::vector<Segment<GRNode*>>& get_node_segment_list() { return _node_segment_list; }
  std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<GRNode*>& get_visited_node_list() { return _visited_node_list; }
  GRNode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_gr_net_ref(GRNet* gr_net_ref) { _gr_net_ref = gr_net_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<GRNode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<GRNode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_path_node_comb(const std::vector<GRNode*>& path_node_comb) { _path_node_comb = path_node_comb; }
  void set_node_segment_list(const std::vector<Segment<GRNode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  void set_forced_routing(const bool forced_routing) { _forced_routing = forced_routing; }
  void set_open_queue(const std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<GRNode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(GRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
  bool isForcedRouting() { return _forced_routing; }
#endif

 private:
  std::vector<GridMap<GRNode>> _layer_node_map;
  std::vector<GRNet> _gr_net_list;
  GRModelStat _gr_model_stat;
#if 1  // astar
  // config
  double _wire_unit = 1;
  double _via_unit = 2;
  // single net
  GRNet* _gr_net_ref = nullptr;
  PlanarRect _routing_region;
  std::vector<std::vector<GRNode*>> _start_node_comb_list;
  std::vector<std::vector<GRNode*>> _end_node_comb_list;
  std::vector<GRNode*> _path_node_comb;
  std::vector<Segment<GRNode*>> _node_segment_list;
  // single path
  bool _forced_routing = false;
  std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost> _open_queue;
  std::vector<GRNode*> _visited_node_list;
  GRNode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt
