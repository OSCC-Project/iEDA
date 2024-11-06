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
 * @file rc_tree.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <variant>
#include <vector>

#include "include/Type.hh"
#include "netlist/Net.hh"
#include "spef/SpefParserRustC.hh"
#include "string/Str.hh"

namespace istagpu {

using ieda::Str;
using ista::CapacitiveUnit;
using ista::Net;
using ista::ResistanceUnit;

struct DelayRcEdge;

/**
 * @brief The rc node for rc tree, represent for capacitance.
 *
 */
struct DelayRcPoint {
  DelayRcPoint() {}
  DelayRcPoint(std::string node_name) : _node_name(node_name) {}
  std::string _node_name;
  float _cap = 0.0;
  float _nload =
      0.0;  //!< The nload is sum of the node cap and downstream node cap.
  float _ndelay = 0.0;  //!< The ndelay is the time from root to this node.
  float _ldelay = 0.0;  //!< The load is sum of the node cap*ndelay and
  //!< downstream node cap*ndelay.
  float _beta = 0.0;
  float _impulse = 0.0;  //!< The delay is the output slew.

  std::size_t _flatten_pos = 0;

  bool _is_update_load = false;
  bool _is_update_delay = false;
  bool _is_update_ldelay = false;
  bool _reserved = false;

  DelayRcPoint* _parent =
      nullptr;  //!< For the tree, the root is the top parent.
  std::vector<DelayRcEdge*> _fanin_edges;  //!< The fanin edge to the rc point.
  std::vector<DelayRcEdge*>
      _fanout_edges;  //!< The fanout edge from the rc point.
};

struct CoupledDelayRcPoint {
  CoupledDelayRcPoint(const std::string& aggressor_node,
                      const std::string& victim_node, double coupled_cap)
      : _local_node(aggressor_node),
        _remote_node(victim_node),
        _coupled_cap(coupled_cap) {}

  std::string _local_node;
  std::string _remote_node;
  float _coupled_cap;
};

/**
 * @brief The rc edge for rc tree, represent for resitance.
 *
 */
struct DelayRcEdge {
  DelayRcEdge() {}
  DelayRcEdge(DelayRcPoint* from, DelayRcPoint* to, float res)
      : _from{from}, _to{to}, _resistance{res} {}
  DelayRcPoint* _from;  // The from node id.
  DelayRcPoint* _to;    // The to node id.

  float _resistance;
};

/**
 * @brief The rc tree of one net.
 *
 */
struct DelayRcNetwork {
  DelayRcPoint* insert_node(const std::string& name, double cap) {
    auto it = _str2nodes.find(name);

    if (it != _str2nodes.end()) {
      if (ista::IsDoubleEqual(it->second->_cap, cap)) {
        return it->second.get();
      } else {
        it->second->_node_name = name;
        it->second->_cap = cap;
        return it->second.get();
      }
    } else {
      auto new_node = std::make_unique<DelayRcPoint>();
      new_node->_node_name = name;
      new_node->_cap = cap;
      _str2nodes[name] = std::move(new_node);

      for (const auto& node : _str2nodes) {
        LOG_INFO << node.second->_node_name;
      }
      return _str2nodes[name].get();
    }
  }

  CoupledDelayRcPoint* insert_node(const std::string& local_node,
                                   const std::string& remote_node,
                                   float coupled_cap) {
    auto coupled_node = std::make_unique<CoupledDelayRcPoint>(
        local_node, remote_node, coupled_cap);
    auto& ret_coupled_node =
        _coupled_nodes.emplace_back(std::move(coupled_node));
    return ret_coupled_node.get();
  }

  DelayRcEdge* insert_edge(const std::string& from, const std::string& to,
                           double res) {
    if (_str2nodes.end() == _str2nodes.find(from)) {
      LOG_INFO_FIRST_N(10) << "spef from node " << from << " is not exist.";
      insert_node(from, 0.0001);
    }
    if (_str2nodes.end() == _str2nodes.find(to)) {
      LOG_INFO_FIRST_N(10) << "spef to node " << to << " is not exist.";
      insert_node(to, 0.0001);
    }

    auto& tail = _str2nodes[from];
    auto& head = _str2nodes[to];

    auto& edge = _edges.emplace_back(
        std::make_unique<DelayRcEdge>(tail.get(), head.get(), res));
    tail->_fanout_edges.push_back(edge.get());
    head->_fanin_edges.push_back(edge.get());
    return edge.get();
  }
  void insert_segment(const std::string& name1, const std::string& name2,
                      double res) {
    insert_edge(name1, name2, res);
    insert_edge(name2, name1, res);
  }
  void sync_nodes() {
    for (auto& pair : _str2nodes) {
      _nodes.push_back(std::move(pair.second));
    }

    _str2nodes.clear();
  }
  DelayRcPoint* rc_node(const std::string& name) {
    for (const auto& node : _nodes) {
      if (node && strcmp(node->_node_name.c_str(), name.c_str()) == 0) {
        return node.get();
      }
    }
    return nullptr;
  }

  DelayRcPoint* _root{nullptr};
  std::vector<std::unique_ptr<DelayRcPoint>> _nodes;
  std::map<std::string, std::unique_ptr<DelayRcPoint>> _str2nodes;
  std::vector<std::unique_ptr<DelayRcEdge>> _edges;
  std::vector<std::unique_ptr<CoupledDelayRcPoint>> _coupled_nodes;

  std::vector<float> _cap_array;
  std::vector<float>
      _load_array;  // TODO(to taosimin), load and delay may be use one array.
  std::vector<float> _resistance_array;
  std::vector<float> _delay_array;
  std::vector<float> _ldelay_array;
  std::vector<float> _beta_array;
  std::vector<float> _impulse_array;
  std::vector<int> _parent_pos_array;
  std::vector<int> _children_pos_array;

  float* _gpu_cap_array = nullptr;
  float* _gpu_load_array = nullptr;
  float* _gpu_resistance_array = nullptr;
  float* _gpu_delay_array = nullptr;
  float* _gpu_ldelay_array = nullptr;
  float* _gpu_beta_array = nullptr;
  float* _gpu_impulse_array = nullptr;
  int* _gpu_parent_pos_array = nullptr;
  int* _gpu_children_pos_array = nullptr;

  std::size_t get_node_num() { return _nodes.size(); }
};

/**
 * @brief The spef common head information.
 *
 */
class DelayRcNetCommonInfo {
 public:
  void set_spef_cap_unit(const std::string& spef_cap_unit) {
    // The unit is 1.0 FF, fix me
    if (Str::contain(spef_cap_unit.c_str(), "1 FF") ||
        Str::contain(spef_cap_unit.c_str(), "1.0 FF")) {
      _spef_cap_unit = CapacitiveUnit::kFF;
    } else {
      _spef_cap_unit = CapacitiveUnit::kPF;
    }
  }
  void set_spef_resistance_unit(const std::string& spef_resistance_unit) {
    // The unit is 1.0 OHM, fix me
    if (Str::contain(spef_resistance_unit.c_str(), "1 OHM") ||
        Str::contain(spef_resistance_unit.c_str(), "1.0 OHM")) {
      _spef_resistance_unit = ResistanceUnit::kOHM;
    } else {
      _spef_resistance_unit = ResistanceUnit::kOHM;
    }
  }
  CapacitiveUnit get_spef_cap_unit() { return _spef_cap_unit; }
  ResistanceUnit get_spef_resistance_unit() { return _spef_resistance_unit; }

 private:
  CapacitiveUnit _spef_cap_unit;
  ResistanceUnit _spef_resistance_unit;
};

/**
 * @brief The rc net wrap for rc tree.
 *
 */
struct DelayRcNet {
  DelayRcNet() {};
  explicit DelayRcNet(Net* net) : _net(net) {}
  static void set_rc_net_common_info(
      std::unique_ptr<DelayRcNetCommonInfo>&& delay_rc_net_common_info) {
    _delay_rc_net_common_info = std::move(delay_rc_net_common_info);
  }

  static DelayRcNetCommonInfo* get_rc_net_common_info() {
    return _delay_rc_net_common_info.get();
  }

  Net* _net;
  DelayRcNetwork _rc_network;
  static std::unique_ptr<DelayRcNetCommonInfo> _delay_rc_net_common_info;
};

#if 1
std::vector<std::vector<DelayRcPoint*>> delay_levelization(
    DelayRcNetwork* rc_network);
void delay_update_point_load(std::vector<std::vector<DelayRcPoint*>>);
void delay_update_point_delay(std::vector<std::vector<DelayRcPoint*>>);
void delay_update_point_ldelay(std::vector<std::vector<DelayRcPoint*>>);
void delay_update_point_impulse(std::vector<std::vector<DelayRcPoint*>>);
void update_rc_timing(DelayRcNet* rc_net);
void make_delay_rct(DelayRcNet* delay_rc_net, RustSpefNet* rust_spef_net);
void update_rc_tree_info(DelayRcNet* delay_rc_net);

#else
float delay_update_point_load(DelayRcPoint* parent, DelayRcPoint* rc_point);
void delay_update_point_load(DelayRcNet* rc_net);

void delay_update_point_delay(DelayRcPoint* parent, DelayRcPoint* rc_net);
void delay_update_point_delay(DelayRcNet* rc_net);

#endif

////////////////////////////////////////////////////

int test();

}  // namespace istagpu
