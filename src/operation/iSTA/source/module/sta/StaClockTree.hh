// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaClockTree.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The clock tree representing the delay between the clock source and the
 * end sink.
 * @version 0.1
 * @date 2023-03-27
 */
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ista {
class StaClock;
class StaClockTreeArc;

/**
 * @brief The class for the arc delay(max/min,rise/fall).
 *
 */
class ModeTransAT {
 public:
  ModeTransAT() = default;
  ModeTransAT(const char* from_name, const char* to_name,
              double max_rise_to_arrive_time, double max_fall_to_arrive_time,
              double min_rise_to_arrive_time, double min_fall_to_arrive_time);
  ~ModeTransAT() = default;

  auto& get_from_name() { return _from_name; }
  void set_from_name(const char* from_name) { _from_name = from_name; }

  auto& get_to_name() { return _to_name; }
  void set_to_name(const char* to_name) { _to_name = to_name; }

  double get_max_rise_to_arrive_time() const {
    return _max_rise_to_arrive_time;
  }
  void set_max_rise_to_arrive_time(double max_rise_to_arrive_time) {
    _max_rise_to_arrive_time = max_rise_to_arrive_time;
  }

  double get_max_fall_to_arrive_time() const {
    return _max_fall_to_arrive_time;
  }
  void set_max_fall_to_arrive_time(double max_fall_to_arrive_time) {
    _max_fall_to_arrive_time = max_fall_to_arrive_time;
  }

  double get_min_rise_to_arrive_time() const {
    return _min_rise_to_arrive_time;
  }
  void set_min_rise_to_arrive_time(double min_rise_to_arrive_time) {
    _min_rise_to_arrive_time = min_rise_to_arrive_time;
  }

  double get_min_fall_to_arrive_time() const {
    return _min_fall_to_arrive_time;
  }
  void set_min_fall_to_arive_time(double min_fall_to_arrive_time) {
    _min_fall_to_arrive_time = min_fall_to_arrive_time;
  }

 private:
  std::string _from_name;
  std::string _to_name;
  double _max_rise_to_arrive_time = 0.0;  //!< The to node arrive time, below.
  double _max_fall_to_arrive_time = 0.0;
  double _min_rise_to_arrive_time = 0.0;
  double _min_fall_to_arrive_time = 0.0;
};

/**
 * @brief The class for clock tree node.
 *
 */
class StaClockTreeNode {
 public:
  using Pin2AT = std::pair<std::string, double>;

  StaClockTreeNode() = default;
  StaClockTreeNode(std::string cell_type, std::string inst_name);
  ~StaClockTreeNode() = default;

  const char* get_cell_type() const { return _cell_type.c_str(); }
  void set_cell_type(const char* cell_type) { _cell_type = cell_type; }

  const char* get_inst_name() const { return _inst_name.c_str(); }
  std::string get_inst_name_str() { return _inst_name; }
  void set_inst_name(const char* inst_name) { _inst_name = inst_name; }

  void addChildNode(StaClockTreeNode* child_node) {
    _child_nodes.emplace_back(child_node);
  }
  auto& get_child_nodes() { return _child_nodes; }

  void addInstArrvieTime(ModeTransAT&& inst_arrive_time) {
    _inst_arrive_times.emplace_back(std::move(inst_arrive_time));
  }
  auto& get_inst_arrive_times() { return _inst_arrive_times; }

  void addFaninArc(StaClockTreeArc* fanin_arc) {
    _fanin_arcs.emplace_back(fanin_arc);
  }
  auto& get_fanin_arcs() { return _fanin_arcs; }

  std::vector<Pin2AT> getInputPinMaxRiseAT();
  std::vector<Pin2AT> getOutputPinMaxRiseAT();

 private:
  std::string _cell_type;
  std::string _inst_name;
  std::vector<ModeTransAT>
      _inst_arrive_times;  // inst internal output arrive time.
  std::vector<StaClockTreeArc*>
      _fanin_arcs;  // fanin arc for get input arrive time.

  std::vector<StaClockTreeNode*> _child_nodes;
};

/**
 * @brief The class for clock tree arc.
 *
 */
class StaClockTreeArc {
 public:
  StaClockTreeArc() = default;
  StaClockTreeArc(StaClockTreeNode* parent_node, StaClockTreeNode* child_node);
  StaClockTreeArc(const char* net_name, StaClockTreeNode* parent_node,
                  StaClockTreeNode* child_node);
  ~StaClockTreeArc() = default;

  const char* get_net_name() const { return _net_name.c_str(); }
  void set_net_name(const char* net_name) { _net_name = net_name; }

  StaClockTreeNode* get_parent_node() const { return _parent_node; }
  void set_parent_node(StaClockTreeNode* parent_node) {
    _parent_node = parent_node;
  }

  StaClockTreeNode* get_child_node() const { return _child_node; }
  void set_child_node(StaClockTreeNode* child_node) {
    _child_node = child_node;
  }

  void set_net_arrive_time(ModeTransAT net_arrive_time) {
    _net_arrive_time = net_arrive_time;
  }
  ModeTransAT get_net_arrive_time() const { return _net_arrive_time; }

 private:
  std::string _net_name;
  StaClockTreeNode* _parent_node;
  StaClockTreeNode* _child_node;
  ModeTransAT _net_arrive_time;
};

/**
 * @brief The class for clock tree.
 *
 */
class StaClockTree {
 public:
  StaClockTree(StaClock* clock, StaClockTreeNode* root_node);
  ~StaClockTree() = default;

  auto* get_clock() { return _clock; }

  StaClockTreeNode* get_root_node() const { return _root_node.get(); }
  void set_root_node(StaClockTreeNode* root_node) {
    _root_node.reset(root_node);
  }

  void addChildNode(StaClockTreeNode* child_node) {
    _child_nodes.emplace_back(child_node);
  }
  auto& get_child_nodes() { return _child_nodes; }
  void clearChildNodes() { _child_nodes.clear(); }
  StaClockTreeNode* findNode(const char* inst_name);
  StaClockTreeNode* findChildNode(const char* inst_name);

  void addChildArc(StaClockTreeArc* child_arc) {
    _child_arcs.emplace_back(child_arc);
  }
  auto& get_child_arcs() { return _child_arcs; }
  StaClockTreeArc* get_child_arc();
  void clearChildfArcs() { _child_arcs.clear(); }
  StaClockTreeArc* findChildArc(StaClockTreeNode* src_node);
  int getChildArcCnt(const char* parent_node);
  std::vector<StaClockTreeNode*> getChildNodes(const char* parent_node);
  std::vector<int> getLevelNodeCnt();
  void getChildNodeCnt(std::vector<StaClockTreeNode*> child_nodes,
                       std::vector<int>& child_node_size);
  void printInstGraphViz(const char* file_path, bool show_port_suffix = true);
  void printInstJson(const char* file_path, bool show_port_suffix = true);

 private:
  StaClock* _clock;  //!< The tree own clock.

  std::unique_ptr<StaClockTreeNode> _root_node;  //!< The root of clock tree.
  std::vector<std::unique_ptr<StaClockTreeNode>>
      _child_nodes;  //!< The child nodes of clock tree.
  std::vector<std::unique_ptr<StaClockTreeArc>>
      _child_arcs;  //!< The child arcs of clock tree.
};

}  // namespace ista