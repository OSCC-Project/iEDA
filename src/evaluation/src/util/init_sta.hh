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
/**
 * @file init_sta.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 * @brief evaluation with iSTA
 */

#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

namespace ista {
enum class AnalysisMode;
}
namespace salt {
class Pin;
}

namespace ivec {
class VecLayout;
}

namespace ieval {

struct TimingNet;

/// @brief The timing node features.
struct TimingNodeFeature {
  using Coord = std::pair<double, double>;
  /// quad data, assume order is max rise, max fall, min rise, min fall.
  using QuadData = std::tuple<double, double, double, double>;

  Coord _node_coord = {0.0, 0.0};
  QuadData _node_slews = {0.0, 0.0, 0.0, 0.0};
  QuadData _node_caps = {0.0, 0.0, 0.0, 0.0};
  /// node arrive time
  QuadData _node_ats = {0.0, 0.0, 0.0, 0.0};  
  /// node required time 
  QuadData _node_rats = {0.0, 0.0, 0.0, 0.0};
  /// node net load delays.
  QuadData _node_net_delays = {0.0, 0.0, 0.0, 0.0};

  unsigned _fanout_num = 1;

  bool _is_input = false;
  bool _is_endpoint = false;
};

/// @brief The timing wire graph for weiguo used.
struct TimingWireNode
{
  std::string _name;  //!< for pin/port name or node id.
  bool _is_pin = false;
  bool _is_port = false;

  TimingNodeFeature _node_feature;
};

/// @brief The timing edge feature.
struct TimingEdgeFeature {
  using QuadData = std::tuple<double, double, double, double>;

  QuadData _edge_delay = {0.0, 0.0, 0.0, 0.0};
  double _edge_resistance = 0.0;
};

struct TimingWireEdge
{
  int64_t _from_node = -1;
  int64_t _to_node = -1;
  bool _is_net_edge = true;

  TimingEdgeFeature _edge_features;
};


/// @brief The timing instance node for wangrui used.
struct TimingInstanceNode {
  std::string _name; //!< instance name
};

struct TimingNetEdge {
  int64_t _from_node = -1;
  int64_t _to_node = -1;
};

template <typename T, typename U>
struct TimingGraph
{
  std::vector<T> _nodes;  //!< each one is a graph node
  std::vector<U> _edges;

 private:
  std::map<std::string, unsigned> _node2index_map;  //!< node name to node index map.
  std::map<std::pair<unsigned, unsigned>, unsigned> _edge2index_map;  //!< edge from/to node index to edge index map.

 public:
  std::optional<unsigned> findNode(std::string& node_name)
  {
    if (_node2index_map.find(node_name) == _node2index_map.end()) {
      return std::nullopt;
    }

    return _node2index_map[node_name];
  }

  unsigned addNode(const T& node)
  {
    _nodes.push_back(node);
    unsigned index = _nodes.size() - 1;
    _node2index_map[node._name] = index;

    return index;
  }
  auto& getNode(unsigned index) { return _nodes[index]; }

  U* findEdge(unsigned wire_from_node_index, unsigned wire_to_node_index)
  {
    auto edge_key = std::make_pair(wire_from_node_index, wire_to_node_index);
    if (_edge2index_map.find(edge_key) == _edge2index_map.end()) {
      return nullptr;
    }

    return &(_edges[_edge2index_map[edge_key]]);
  }

  U& addEdge(unsigned wire_from_node_index, unsigned wire_to_node_index)
  {
    U wire_graph_edge;

    wire_graph_edge._from_node = wire_from_node_index;
    wire_graph_edge._to_node = wire_to_node_index;

    auto edge_key = std::make_pair(wire_from_node_index, wire_to_node_index);
    if (_edge2index_map.find(edge_key) != _edge2index_map.end())
    {
      return _edges[_edge2index_map[edge_key]];
    }

    _edge2index_map[edge_key] = _edges.size();

    return _edges.emplace_back(std::move(wire_graph_edge));
  }
};


/**
 * @brief The timing wire graph for weiguo used.
 */
struct TimingWireGraph : public TimingGraph<TimingWireNode, TimingWireEdge> {

};

/**
 * @brief The timing instance graph for wangrui used.
 */
struct TimingInstanceGraph : public TimingGraph<TimingInstanceNode, TimingNetEdge> {

};

/// @brief  save timing graph to yaml file.
/// @param timing_wire_graph
/// @param yaml_file_name
void SaveTimingGraph(const TimingWireGraph& timing_wire_graph, const std::string& json_file_name);

/// @brief restore timing graph from yaml file.
TimingWireGraph RestoreTimingGraph(const std::string& yaml_file_name);

/// @brief save timing instance graph to yaml file.
void SaveTimingInstanceGraph(const TimingInstanceGraph& timing_instance_graph, const std::string& json_file_name);
/// @brief restore timing instance graph from yaml file.
TimingInstanceGraph RestoreTimingInstanceGraph(const std::string& yaml_file_name);

class InitSTA
{
 public:
  InitSTA() = default;
  ~InitSTA();
  static InitSTA* getInst();
  static void destroyInst();

  void runSTA();
  void runVecSTA(ivec::VecLayout* vec_layout, std::string work_dir);
  void evalTiming(const std::string& routing_type, const bool& rt_done = false);

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> getTiming() const { return _timing; }
  std::map<std::string, std::map<std::string, double>> getPower() const { return _power; }

  std::map<std::string, std::unordered_map<std::string, double>> getNetPower() const { return _net_power; }

  double getEarlySlack(const std::string& pin_name) const;
  double getLateSlack(const std::string& pin_name) const;
  double getArrivalEarlyTime(const std::string& pin_name) const;
  double getArrivalLateTime(const std::string& pin_name) const;
  double getRequiredEarlyTime(const std::string& pin_name) const;
  double getRequiredLateTime(const std::string& pin_name) const;
  double reportWNS(const char* clock_name, ista::AnalysisMode mode);
  double reportTNS(const char* clock_name, ista::AnalysisMode mode);

  // for net R、C、slew、delay power.
  double getNetResistance(const std::string& net_name) const;
  double getNetCapacitance(const std::string& net_name) const;
  double getNetSlew(const std::string& net_name) const;
  std::map<std::string, double> getAllNodesSlew(const std::string& net_name) const;
  double getNetDelay(const std::string& net_name) const;
  std::pair<double, double> getNetToggleAndVoltage(const std::string& net_name) const;
  double getNetPower(const std::string& net_name) const;

  // for wire R、C、slew、delay power.
  double getWireResistance(const std::string& net_name, const std::string& wire_node_name) const;
  double getWireCapacitance(const std::string& net_name, const std::string& wire_node_name) const;
  double getWireDelay(const std::string& net_name, const std::string& wire_node_name) const;
  // double getWirePower(const std::string& net_name, const std::string& wire_node_name) const;
  TimingWireGraph getTimingWireGraph();
  TimingInstanceGraph getTimingInstanceGraph();

  bool getRcNet(const std::string& net_name);

  void buildRCTree(const std::string& routing_type);
  void buildVecRCTree(ivec::VecLayout* vec_layout, std::string work_dir);
  void updateTiming(const std::vector<TimingNet*>& timing_net_list, int32_t dbu_unit);
  void updateTiming(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list, const int& propagation_level,
                    int32_t dbu_unit);

  bool isClockNet(const std::string& net_name) const;

  std::map<int, double> patchTimingMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch);
  std::map<int, double> patchPowerMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch);
  std::map<int, double> patchIRDropMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch);

 private:
  void leaglization(const std::vector<std::shared_ptr<salt::Pin>>& pins);
  void initStaEngine();
  void callRT(const std::string& routing_type);

  void initPowerEngine();
  void updateResult(const std::string& routing_type);

  static InitSTA* _init_sta;

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> _timing;
  std::map<std::string, std::map<std::string, double>> _power;
  std::map<std::string, std::unordered_map<std::string, double>> _net_power;
};

}  // namespace ieval