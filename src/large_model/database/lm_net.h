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
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <string>
#include <vector>

#include "lm_node.h"

namespace ilm {

struct LmNetWireFeature
{
  int wire_width = 0;
  int wire_len = 0;
  int drc_num = 0;
  double R = 0.0;
  double C = 0.0;
  double power = 0.0;
  double delay = 0.0;
  double slew = 0.0;
  double congestion = 0.0;
  double wire_density = 0.0;
  std::vector<std::string> drc_type = {};
};

struct LmNetFeature
{
  int llx = 0;
  int lly = 0;
  int urx = 0;
  int ury = 0;
  int wire_len = 0;
  int via_num = 0;
  int drc_num = 0;
  double R = 0.0;
  double C = 0.0;
  double power = 0.0;
  double delay = 0.0;
  double slew = 0.0;
  int fanout = 0;
  int aspect_ratio = 0;
  int64_t width = 0;
  int64_t height = 0;
  int64_t area = 0;
  float l_ness = 0.0;
  std::vector<std::string> drc_type = {};
  int64_t volume = 0;
  std::vector<int> layer_ratio = {};
  int rsmt = 0;
};

static int64_t wire_id_index = 0;

class LmNetWire
{
 public:
  LmNetWire(LmNode* node1 = nullptr, LmNode* node2 = nullptr, int id = -1)
  {
    _node_pair = std::make_pair(node1, node2);
    _id = id == -1 ? wire_id_index++ : id;
  };
  ~LmNetWire() { _paths.clear(); }

  // getter
  int64_t get_id() { return _id; }
  std::pair<LmNode*, LmNode*>& get_connected_nodes() { return _node_pair; }
  std::vector<std::pair<LmNode*, LmNode*>>& get_paths() { return _paths; }
  LmNetWireFeature* get_feature(bool b_create = false) { return &_feature; }
  std::map<int, std::set<int>>& get_patchs() { return _patchs; }

  // setter
  void set_start(LmNode* node) { _node_pair.first = node; }
  void set_end(LmNode* node) { _node_pair.second = node; }

  void add_path(LmNode* node1, LmNode* node2);

  // operator
  void addPatch(int patch_id, int layer_id);

 private:
  int64_t _id = -1;
  std::pair<LmNode*, LmNode*> _node_pair;
  std::vector<std::pair<LmNode*, LmNode*>> _paths;

  LmNetWireFeature _feature;

  std::map<int, std::set<int>> _patchs;
};

struct LmPin
{
  int pin_id = -1;
  std::string pin_name = "";
  std::string instance_name = "";
  bool is_driver = false;
};


struct NetRoutingPoint {
  int x;
  int y;
  int layer_id;
};

struct NetRoutingVertex {
  size_t id;
  bool is_pin;
  bool is_driver_pin;
  NetRoutingPoint point;
};

struct NetRoutingEdge {
  size_t source_id;
  size_t target_id;

  std::vector<NetRoutingPoint> path;
};

struct NetRoutingGraph {
  std::vector<NetRoutingVertex> vertices;
  std::vector<NetRoutingEdge> edges;
};


class LmNet
{
 public:
  LmNet(int net_id) : _net_id(net_id) {}
  ~LmNet() {}

  // getter
  int get_net_id() { return _net_id; }
  std::vector<LmNetWire>& get_wires() { return _wires; }
  std::vector<int>& get_pin_ids() { return _pin_ids; }
  LmNetFeature* get_feature(bool b_create = false);
  std::map<int, LmPin>& get_pin_list() { return _pin_list; }
  NetRoutingGraph get_routing_graph() { return _routing_graph; }
  // setter
  void set_net_id(int net_id) { _net_id = net_id; }
  void set_routing_graph(const NetRoutingGraph& routing_graph) { _routing_graph = routing_graph; }

  // operator
  void addWire(LmNetWire wire);
  void clearWire() { _wires.clear(); }
  void addPinId(int id) { _pin_ids.push_back(id); }
  void addPin(int id, LmPin pin) { _pin_list.insert(std::make_pair(id, pin)); }
  LmNetWire* findWire(int64_t wire_id);

 private:
  int _net_id = -1;
  std::vector<LmNetWire> _wires;
  std::vector<int> _pin_ids;
  std::map<int, LmPin> _pin_list;
  LmNetFeature* _feature = nullptr;
  NetRoutingGraph _routing_graph;
};

class LmGraph
{
 public:
  LmGraph() {}
  ~LmGraph() {}

  // getter
  std::map<int, LmNet>& get_net_map() { return _net_map; }
  LmNet* get_net(int net_id);

  // setter
  LmNet* addNet(int net_id);
  void add_net_wire(int net_id, LmNetWire wire);

  // operator

 private:
  std::map<int, LmNet> _net_map;
};

}  // namespace ilm
