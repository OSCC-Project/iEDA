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
 * @project		vectorization
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

#include "vec_node.h"

namespace ivec {

struct VecNetWireFeature
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

struct VecPlaceFeature
{
  int pin_num = 0;
  int aspect_ratio = 0;
  int64_t width = 0;
  int64_t height = 0;
  int64_t area = 0;
  float l_ness = 0.0;
  int64_t hpwl = 0;
  int64_t rsmt = 0;
};

struct VecNetFeature
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
  int aspect_ratio = 0;
  int64_t width = 0;
  int64_t height = 0;
  int64_t area = 0;
  int64_t volume = 0;
  std::vector<std::string> drc_type = {};
  std::vector<float> layer_ratio = {};
  VecPlaceFeature place_feature;
};

static int64_t wire_id_index = 0;

class VecNetWire
{
 public:
  VecNetWire(VecNode* node1 = nullptr, VecNode* node2 = nullptr, int id = -1)
  {
    _node_pair = std::make_pair(node1, node2);
    _id = id == -1 ? wire_id_index++ : id;
  };
  ~VecNetWire() { _paths.clear(); }

  // getter
  int64_t get_id() { return _id; }
  std::pair<VecNode*, VecNode*>& get_connected_nodes() { return _node_pair; }
  std::vector<std::pair<VecNode*, VecNode*>>& get_paths() { return _paths; }
  VecNetWireFeature* get_feature(bool b_create = false) { return &_feature; }
  std::map<int, std::set<int>>& get_patchs() { return _patchs; }

  // setter
  void set_start(VecNode* node) { _node_pair.first = node; }
  void set_end(VecNode* node) { _node_pair.second = node; }

  void add_path(VecNode* node1, VecNode* node2);

  // operator
  void addPatch(int patch_id, int layer_id);

 private:
  int64_t _id = -1;
  std::pair<VecNode*, VecNode*> _node_pair;
  std::vector<std::pair<VecNode*, VecNode*>> _paths;

  VecNetWireFeature _feature;

  std::map<int, std::set<int>> _patchs;
};

struct VecPin
{
  int pin_id = -1;
  std::string pin_name = "";
  std::string instance_name = "";
  bool is_driver = false;
};

struct NetRoutingPoint
{
  int x;
  int y;
  int layer_id;
};

struct NetRoutingVertex
{
  size_t id;
  bool is_pin;
  bool is_driver_pin;
  NetRoutingPoint point;
};

struct NetRoutingEdge
{
  size_t source_id;
  size_t target_id;

  std::vector<NetRoutingPoint> path;
};

struct NetRoutingGraph
{
  std::vector<NetRoutingVertex> vertices;
  std::vector<NetRoutingEdge> edges;
};

class VecNet
{
 public:
  VecNet(int net_id) : _net_id(net_id) {}
  ~VecNet() {}

  // getter
  int get_net_id() { return _net_id; }
  std::string get_net_name() { return _net_name; }
  std::vector<VecNetWire>& get_wires() { return _wires; }
  std::vector<int>& get_pin_ids() { return _pin_ids; }
  VecNetFeature* get_feature(bool b_create = false);
  std::map<int, VecPin>& get_pin_list() { return _pin_list; }
  NetRoutingGraph get_routing_graph() { return _routing_graph; }
  // setter
  void set_net_id(int net_id) { _net_id = net_id; }
  void set_net_name(std::string name) { _net_name = name; }
  void set_routing_graph(const NetRoutingGraph& routing_graph) { _routing_graph = routing_graph; }

  // operator
  void addWire(VecNetWire wire);
  void clearWire() { _wires.clear(); }
  void addPinId(int id) { _pin_ids.push_back(id); }
  void addPin(int id, VecPin pin) { _pin_list.insert(std::make_pair(id, pin)); }
  VecNetWire* findWire(int64_t wire_id);

 private:
  int _net_id = -1;
  std::string _net_name;
  std::vector<VecNetWire> _wires;
  std::vector<int> _pin_ids;
  std::map<int, VecPin> _pin_list;
  VecNetFeature* _feature = nullptr;
  NetRoutingGraph _routing_graph;
};

class VecGraph
{
 public:
  VecGraph() {}
  ~VecGraph() {}

  // getter
  std::map<int, VecNet>& get_net_map() { return _net_map; }
  VecNet* get_net(int net_id);

  // setter
  VecNet* addNet(int net_id);
  void add_net_wire(int net_id, VecNetWire wire);

  // operator

 private:
  std::map<int, VecNet> _net_map;
};

}  // namespace ivec
