// ***************************************************************************************
// MIT License
//
// Copyright (c) 2018-2021 Tsung-Wei Huang and Martin D. F. Wong
//
// The University of Utah, UT, USA
//
// The University of Illinois at Urbana-Champaign, IL, USA
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ***************************************************************************************
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
 * @file DelayCalc.cc
 * @author LH (liuh0326@163.com)
 * @brief The file implement traditional elmore calc algorithm.
 * @version 0.1
 * @date 2021-01-27
 */

#include <numeric>
#include <queue>
#include <utility>

#include "log/Log.hh"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Net.hh"
#include "netlist/Pin.hh"
#include "netlist/Port.hh"

namespace ista {

const int THREAD_PER_BLOCK_NUM = 512;

std::unique_ptr<RCNetCommonInfo> RcNet::_rc_net_common_info;

RctNode::RctNode(std::string&& name)
    : _name{std::move(name)},
      _is_update_load(0),
      _is_update_delay(0),
      _is_update_ldelay(0),
      _is_update_delay_ecm(0),
      _is_update_m2(0),
      _is_update_mc(0),
      _is_update_m2_c(0),
      _is_update_mc_c(0),
      _is_update_ceff(0),
      _is_update_response(0),
      _is_tranverse(0),
      _is_visited(0),
      _is_visited_ecm(0),
      _is_root(0),
      _reserved(0) {}

void RctNode::calNodePIModel() {
  if (IsDoubleEqual(_moments.y2, 0.0) || IsDoubleEqual(_moments.y3, 0.0)) {
    return;
  }

  double y1 = _moments.y1;
  double y2 = _moments.y2;
  double y3 = _moments.y3;
  double C1 = pow(y2, 2) / y3;
  double C2 = y1 - pow(y2, 2) / y3;
  double R = -pow(y3, 2) / pow(y2, 3);

  _pi.C_near = C2;
  _pi.R = R;
  _pi.C_far = C1;
}

double RctNode::nodeLoad(AnalysisMode mode, TransType trans_type) {
  return _nload[ModeTransPair(mode, trans_type)];
}

double RctNode::cap() const { return _obj ? _obj->cap() + _cap : _cap; }

double RctNode::cap(AnalysisMode mode, TransType trans_type) {
  return _obj ? _obj->cap(mode, trans_type) +
                    _ncap[ModeTransPair(mode, trans_type)]
              : _ncap[ModeTransPair(mode, trans_type)];
}

void RctNode::setCap(double cap) {
  _cap = cap;
  FOREACH_MODE_TRANS(mode, trans) { _ncap[ModeTransPair(mode, trans)] = cap; }
}

void RctNode::incrCap(double cap) {
  _cap += cap;
  FOREACH_MODE_TRANS(mode, trans) { _ncap[ModeTransPair(mode, trans)] += cap; }
}

double RctNode::delay(AnalysisMode mode, TransType trans_type) {
  return _ndelay[ModeTransPair(mode, trans_type)];
}

double RctNode::slew(AnalysisMode mode, TransType trans_type,
                     double input_slew) {
  auto si = input_slew;
  return si < 0.0
             ? -std::sqrt(si * si + _impulse[ModeTransPair(mode, trans_type)])
             : std::sqrt(si * si + _impulse[ModeTransPair(mode, trans_type)]);
}

RctEdge::RctEdge(RctNode& from, RctNode& to, double res)
    : _from{from},
      _to{to},
      _res{res},
      _is_break(0),
      _is_visited(0),
      _is_in_order(0),
      _reserved(0) {
  LOG_FATAL_IF(to.get_name().empty());
}

RctNode* RcTree::rcNode(const std::string& name) {
  if (auto itr = _str2nodes.find(name); itr != _str2nodes.end()) {
    return &(itr->second);
  }

  return nullptr;
}

RctNode* RcTree::node(const std::string& name) {
  if (const auto itr = _str2nodes.find(name); itr != _str2nodes.end()) {
    return &(itr->second);
  }

  return nullptr;
}

/**
 * @brief insert ground cap node.
 *
 * @param name
 * @param cap
 * @return RctNode*
 */
RctNode* RcTree::insertNode(const std::string& name, double cap) {
  if (_str2nodes.find(name) != _str2nodes.end() &&
      IsDoubleEqual(_str2nodes[name]._cap, cap)) {
    return &_str2nodes[name];
  }

  auto& node = _str2nodes[name];
  node._name = name;
  node._cap = cap;

  FOREACH_MODE_TRANS(mode, trans) {
    node._ncap[ModeTransPair(mode, trans)] = cap;
  }

  return &node;
}

/**
 * @brief insert coupled cap node.
 *
 * @param local_node
 * @param remote_node
 * @param coupled_cap
 * @return CoupledRcNode*
 */
CoupledRcNode* RcTree::insertNode(const std::string& local_node,
                                  const std::string& remote_node,
                                  double coupled_cap) {
  auto& coupled_node =
      _coupled_nodes.emplace_back(local_node, remote_node, coupled_cap);
  return &coupled_node;
}

/**
 * @brief insert resistance edge.
 *
 * @param from
 * @param to
 * @param res
 * @return RctEdge*
 */
RctEdge* RcTree::insertEdge(const std::string& from, const std::string& to,
                            double res) {
  if (_str2nodes.end() == _str2nodes.find(from)) {
    LOG_INFO_FIRST_N(10) << "spef from node " << from << " is not exist.";
    insertNode(from, 0.0001);
  }

  if (_str2nodes.end() == _str2nodes.find(to)) {
    LOG_INFO_FIRST_N(10) << "spef to node " << to << " is not exist.";
    insertNode(to, 0.0001);
  }

  auto& tail = _str2nodes[from];
  auto& head = _str2nodes[to];

  auto& edge = _edges.emplace_back(tail, head, res);

  tail._fanout.push_back(&edge);
  head._fanin.push_back(&edge);
  return &edge;
}

RctEdge* RcTree::insertEdge(RctNode* from_node, RctNode* to_node, double res,
                            bool in_order) {
  auto& edge = _edges.emplace_back(*from_node, *to_node, res);
  edge.set_is_in_order(in_order);
  if (from_node == to_node) {
    // self loop need break;
    edge.set_is_break();
  }

  from_node->_fanout.push_back(&edge);
  to_node->_fanin.push_back(&edge);
  return &edge;
}

void RcTree::insertSegment(const std::string& name1, const std::string& name2,
                           double res) {
  insertEdge(name1, name2, res);
  insertEdge(name2, name1, res);
}

/**
 * @brief init the rc tree data.
 *
 */
void RcTree::initData() {
  auto init_zero_value = [](auto& the_map) {
    for (auto it = the_map.begin(); it != the_map.end(); it++) {
      it->second = 0.0;
    }
  };

  for (auto& kvp : _str2nodes) {
    kvp.second._load = 0.0;
    kvp.second._res = 0.0;
    kvp.second._delay = 0.0;

    kvp.second._is_update_load = 0;
    kvp.second._is_update_delay = 0;
    kvp.second._is_update_ldelay = 0;
    kvp.second._is_update_response = 0;

    init_zero_value(kvp.second._ures);
    init_zero_value(kvp.second._nload);
    init_zero_value(kvp.second._beta);
    init_zero_value(kvp.second._ndelay);
    init_zero_value(kvp.second._ldelay);
    init_zero_value(kvp.second._impulse);
  }
}

/**
 * @brief init node moment.
 *
 */
void RcTree::initMoment() {
  WaveformApproximation wave_form;
  int load_nodes_pin_cap_sum = 0;
  wave_form.reduceRCTreeToPIModel(_root, load_nodes_pin_cap_sum);
}

/**
 * @brief calculate and update the each node's load of a rctree
 *
 * @param parent
 * @param from
 *
 */
void RcTree::updateLoad(RctNode* parent, RctNode* from) {
  if (from->isUpdateLoad()) {
    LOG_ERROR << "found loop in rc tree " << from->get_name();
    // printGraphViz();
    return;
  }

  from->set_is_update_load(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      updateLoad(from, &to);

      from->_load += to._load;

      FOREACH_MODE_TRANS(mode, trans) {
        from->_nload[ModeTransPair(mode, trans)] +=
            to._nload[ModeTransPair(mode, trans)];
      }
    }
  }
  from->_load += from->cap();

  FOREACH_MODE_TRANS(mode, trans) {
    from->_nload[ModeTransPair(mode, trans)] += from->cap(mode, trans);
  }
}

/**
 * @brief upadate the delay from net root to each node.
 *
 * @param parent
 * @param from
 */
void RcTree::updateDelay(RctNode* parent, RctNode* from) {
  if (from->isUpdateDelay()) {
    return;
  }

  from->set_is_update_delay(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      to._res = from->_res + e->_res;
      to._delay = from->_delay + e->_res * to._load;
      FOREACH_MODE_TRANS(mode, trans) {
        to._ndelay[ModeTransPair(mode, trans)] =
            from->_ndelay[ModeTransPair(mode, trans)] +
            e->_res * to._nload[ModeTransPair(mode, trans)];

        // Update the upstream resistance.
        to._ures[ModeTransPair(mode, trans)] =
            from->_ures[ModeTransPair(mode, trans)] + e->_res;
      }
      updateDelay(from, &to);
    }
  }
}

// Procedure: _update_ldelay
// Compute the load delay of each rctree node along the downstream traversal of
// the rctree.
void RcTree::updateLDelay(RctNode* parent, RctNode* from) {
  if (from->isUpdateLdelay()) {
    return;
  }

  from->set_is_update_Ldelay(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      updateLDelay(from, &to);
      FOREACH_MODE_TRANS(mode, trans) {
        from->_ldelay[ModeTransPair(mode, trans)] +=
            to._ldelay[ModeTransPair(mode, trans)];
      }
    }
  }

  FOREACH_MODE_TRANS(mode, trans) {
    from->_ldelay[ModeTransPair(mode, trans)] +=
        from->cap(mode, trans) * from->_ndelay[ModeTransPair(mode, trans)];
  }
}

// Procedure: _update_response
// Compute the impulse and second moment of the input response for each rctree
// node.
void RcTree::updateResponse(RctNode* parent, RctNode* from) {
  if (from->isUpdateResponse()) {
    return;
  }

  from->set_is_update_response(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      FOREACH_MODE_TRANS(mode, trans) {
        to._beta[ModeTransPair(mode, trans)] =
            from->_beta[ModeTransPair(mode, trans)] +
            e->_res * to._ldelay[ModeTransPair(mode, trans)];
      }
      updateResponse(from, &to);
    }
  }

  FOREACH_MODE_TRANS(mode, trans) {
    from->_impulse[ModeTransPair(mode, trans)] =
        2.0 * from->_beta[ModeTransPair(mode, trans)] -
        std::pow(from->_ndelay[ModeTransPair(mode, trans)], 2);
  }
}

void RcTree::updateDelayECM(RctNode* parent, RctNode* from) {
  if (from->isUpdateDelayECM()) {
    return;
  }

  from->set_is_update_delay_ecm(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      to._delay_ecm = from->_delay_ecm + e->_res * to.updateCeff();

      updateDelayECM(from, &to);
    }
  }
}

void RcTree::updateMC(RctNode* parent, RctNode* from) {
  if (from->isUpdateMC()) {
    return;
  }

  from->set_is_update_mc(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      updateMC(from, &to);

      from->_mc += to._mc;
    }
  }
  from->_mc += from->_delay * from->cap();
}

/**
 * @brief update mc for modify D2M.
 *
 * @param parent
 * @param from
 */
void RcTree::updateMCC(RctNode* parent, RctNode* from) {
  if (from->isUpdateMCC()) {
    return;
  }

  from->set_is_update_mc_c(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      updateMCC(from, &to);

      from->_mc_c += to._mc_c;
    }
  }
  from->_mc_c += from->_delay_ecm * from->cap();
}

void RcTree::updateM2(RctNode* parent, RctNode* from) {
  if (from->isUpdateM2()) {
    return;
  }

  from->set_is_update_delay(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      to._m2 = from->_m2 + e->_res * to._mc;

      updateM2(from, &to);
    }
  }
}

/**
 * @brief update D2M changed.
 *
 * @param parent
 * @param from
 */
void RcTree::updateM2C(RctNode* parent, RctNode* from) {
  if (from->isUpdateM2C()) {
    return;
  }

  from->set_is_update_m2_c(true);

  for (auto* e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      to._m2_c = from->_m2_c + e->_res * to._mc_c;

      updateM2C(from, &to);
    }
  }
}

/**
 * @brief calculate and update the load of each node,calculate and update the
 * delay from net root to each node.
 */
void RcTree::updateRcTiming() {
  if (!_root) {
    LOG_ERROR << "RCTree root can not found";
    return;
  }

  initData();

#if CUDA_DELAY
  levelizeRcTree();
  applyDelayDataToArray();
#else
  updateLoad(nullptr, _root);
  updateDelay(nullptr, _root);
  updateLDelay(nullptr, _root);
  updateResponse(nullptr, _root);
#endif

  if (c_print_delay_yaml) {
    updateMC(nullptr, _root);
    updateM2(nullptr, _root);

    initMoment();
    updateDelayECM(nullptr, _root);
    updateMCC(nullptr, _root);
    updateM2C(nullptr, _root);
  }

  // printGraphViz();
}

double RcTree::delay(const std::string& name) {
  auto itr = _str2nodes.find(name);
  if (itr == _str2nodes.end()) {
    LOG_FATAL << "RCTree node " << name << " can not found." << std::endl;
  }
  return itr->second.delay();
}

double RcTree::delay(const std::string& name, AnalysisMode mode,
                     TransType trans_type) {
  auto itr = _str2nodes.find(name);
  if (itr == _str2nodes.end()) {
    LOG_FATAL << "RCTree node " << name << " can not found." << std::endl;
  }
  return itr->second._ndelay[ModeTransPair(mode, trans_type)];
}

double RcTree::slew(const std::string& name, AnalysisMode mode,
                    TransType trans_type, double input_slew) {
  auto itr = _str2nodes.find(name);
  if (itr == _str2nodes.end()) {
    LOG_FATAL << "RCTree node " << name << " can not found." << std::endl;
  }
  return itr->second.slew(mode, trans_type, input_slew);
}

/**
 * @brief get wire topo of arriving to node.
 *
 * @param to_node_name
 * @return std::vector<RctEdge*>
 */
std::vector<RctEdge*> RcTree::getWireTopo(const char* to_node_name) {
  std::vector<RctEdge*> wire_topo;

  std::function<unsigned(RctNode*, RctNode*)> get_topo_edge =
      [&get_topo_edge, this, &wire_topo, to_node_name](
          RctNode* parent_node, RctNode* src_node) -> unsigned {
    auto& fanout_edges = src_node->get_fanout();
    for (auto* fanout_edge : fanout_edges) {
      auto& snk_node = fanout_edge->_to;
      if (fanout_edge->isBreak() || &snk_node == parent_node) {
        continue;
      }

      if (snk_node.get_name() == to_node_name) {
        wire_topo.push_back(fanout_edge);
        return 1;
      }

      if (get_topo_edge(src_node, &snk_node)) {
        wire_topo.push_back(fanout_edge);
        return 1;
      }
    }

    return 0;
  };

  get_topo_edge(nullptr, _root);

  LOG_FATAL_IF((wire_topo.empty() ||
                (*wire_topo.begin())->_to.get_name() != to_node_name))
      << "not found to node name " << to_node_name;

  return wire_topo;
}

/**
 * @brief get rc tree all node slew.
 *
 * @param driver_slew
 * @return std::map<std::string, double>
 */
std::map<std::string, double> RcTree::getAllNodeSlew(double driver_slew,
                                                     AnalysisMode analysis_mode,
                                                     TransType trans_type) {
  std::map<std::string, double> all_node_slews;
  auto* rc_root = get_root();

  all_node_slews[rc_root->get_name()] = driver_slew;

  std::function<void(RctNode*, RctNode*)> get_snk_slew =
      [&get_snk_slew, this, driver_slew, analysis_mode, trans_type,
       &all_node_slews](RctNode* parent_node, RctNode* src_node) {
        auto& fanout_edges = src_node->get_fanout();
        for (auto* fanout_edge : fanout_edges) {
          auto& snk_node = fanout_edge->get_to();
          if (fanout_edge->isBreak() || &snk_node == parent_node) {
            continue;
          }

          auto snk_slew =
              snk_node.slew(analysis_mode, trans_type, NS_TO_PS(driver_slew));
          all_node_slews[snk_node.get_name()] = PS_TO_NS(snk_slew);

          get_snk_slew(src_node, &snk_node);
        }
      };

  get_snk_slew(nullptr, rc_root);

  return all_node_slews;
}

/**
 * @brief Print the rc tree to graphviz dot file format.
 *
 */
void RcTree::printGraphViz() {
  LOG_INFO << "dump graph dotviz start";

  std::ofstream dot_file;
  dot_file.open("./tree_gpu_gloabl.dot", std::ios::app);

  auto replace_colon = [](const std::string& s) {
    std::string modified = s;
    std::replace(modified.begin(), modified.end(), ':', '_');
    return modified;
  };

  dot_file << "digraph tree" << replace_colon(_root->get_name()).c_str()
           << "{\n";

  for (auto& edge : _edges) {
    // if (!edge.isInOrder()) {
    //   continue;
    // }
    auto from_name = edge._from.get_name();
    auto to_name = edge._to.get_name();
    ModeTransPair mode_trans = {AnalysisMode::kMax, TransType::kRise};

    dot_file << Str::printf(
        "p%p[label=\"%s load %f nload %f delay %f  ndelay %f  ures %f ldelay "
        "%f beta %f impulse %f\" ]\n",
        &edge._from, replace_colon(from_name).c_str(), edge._from._load,
        edge._from._nload[mode_trans], edge._from._delay,
        edge._from._ndelay[mode_trans], edge._from._ures[mode_trans],
        edge._from._ldelay[mode_trans], edge._from._beta[mode_trans],
        edge._from._impulse[mode_trans]);

    dot_file << Str::printf("p%p", &edge._from) << " -> "
             << Str::printf("p%p", &edge._to)
             << Str::printf("[label=\"res %f\" ]", edge.get_res()) << "\n";

    dot_file << Str::printf(
        "p%p[label=\"%s load %f nload %f delay %f  ndelay %f  ures %f ldelay "
        "%f beta %f impulse %f\" ]\n",
        &edge._to, replace_colon(to_name).c_str(), edge._to._load,
        edge._to._nload[mode_trans], edge._to._delay,
        edge._to._ndelay[mode_trans], edge._to._ures[mode_trans],
        edge._to._ldelay[mode_trans], edge._to._beta[mode_trans],
        edge._to._impulse[mode_trans]);
  }

  dot_file << "}\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";
}

void RcTree::levelizeRcTree(std::queue<RctNode*> bfs_queue) {
  std::queue<RctNode*> next_bfs_queue;
  std::vector<RctNode*> points;

  while (!bfs_queue.empty()) {
    auto* rc_node = bfs_queue.front();
    bfs_queue.pop();

    for (auto* fanout_edge : rc_node->get_fanout()) {
      if ((&fanout_edge->get_to()) != rc_node->get_parent()) {
        fanout_edge->get_to().set_parent(rc_node);
        next_bfs_queue.push(&(fanout_edge->get_to()));
        points.emplace_back(&(fanout_edge->get_to()));
      }
    }
  }

  if (!points.empty()) {
    _level_to_points.emplace_back(std::move(points));
  }

  if (!next_bfs_queue.empty()) {
    levelizeRcTree(next_bfs_queue);
  }
}

void RcTree::levelizeRcTree() {
  std::vector<RctNode*> points{_root};
  _level_to_points.emplace_back(points);

  std::queue<RctNode*> bfs_queue;
  bfs_queue.push(_root);

  levelizeRcTree(std::move(bfs_queue));
}

inline ModeTransIndex mapToModeTransIndex(AnalysisMode mode, TransType type) {
  if (mode == AnalysisMode::kMax) {
    if (type == TransType::kRise) {
      return ModeTransIndex::kMaxRise;
    } else if (type == TransType::kFall) {
      return ModeTransIndex::kMaxFall;
    }
  } else if (mode == AnalysisMode::kMin) {
    if (type == TransType::kRise) {
      return ModeTransIndex::kMinRise;
    } else if (type == TransType::kFall) {
      return ModeTransIndex::kMinFall;
    }
  }
  throw std::invalid_argument("Invalid AnalysisMode or TransType combination");
}

void RcTree::applyDelayDataToArray() {
  int node_num = numNodes();
  std::vector<float> cap_array;
  std::vector<float> ncap_array;
  std::vector<float> load_array(node_num, 0);
  std::vector<float> nload_array(node_num * 4, 0);
  std::vector<int> parent_pos_array(node_num, 0);
  // children use start and end pair to mark position.
  std::vector<int> children_pos_array(node_num * 2, 0);
  // resistance record the parent resistance with the children.The first one is
  // root, resistance is 0.
  std::vector<float> res_array{0.0};
  std::vector<float> delay_array(node_num, 0);
  std::vector<float> ndelay_array(node_num * 4, 0);
  std::vector<float> ures_array(node_num * 4, 0);
  std::vector<float> ldelay_array(node_num * 4, 0);
  std::vector<float> beta_array(node_num * 4, 0);
  std::vector<float> impulse_array(node_num * 4, 0);

  int flatten_pos = 0;
  for (auto& points : _level_to_points) {
    for (auto* rc_node : points) {
      rc_node->set_flatten_pos(flatten_pos);
      cap_array.emplace_back(rc_node->cap());

      std::vector<float> one_node_ncap(4, 0);

      FOREACH_MODE_TRANS(mode, trans) {
        ModeTransIndex index = mapToModeTransIndex(mode, trans);
        one_node_ncap[static_cast<int>(index)] = rc_node->cap(mode, trans);
      }

      ncap_array.insert(ncap_array.end(), one_node_ncap.begin(),
                        one_node_ncap.end());

      if (rc_node->get_parent()) {
        auto found_edge = findEdge(*(rc_node->get_parent()), *rc_node);
        assert(found_edge.has_value());
        res_array.emplace_back((*found_edge)->get_res());

        int parent_pos = rc_node->get_parent()->get_flatten_pos();
        if (flatten_pos >= node_num) {
          LOG_FATAL << "flatten pos " << flatten_pos
                    << " is larger than node num " << node_num;
        }
        parent_pos_array[flatten_pos] = parent_pos;

        if (children_pos_array[parent_pos * 2] == 0) {
          children_pos_array[parent_pos * 2] = flatten_pos;
        } else {
          children_pos_array[parent_pos * 2 + 1] = flatten_pos;
        }
      }

      ++flatten_pos;
    }
  }

  std::swap(_cap_array, cap_array);
  std::swap(_ncap_array, ncap_array);
  std::swap(_res_array, res_array);
  std::swap(_parent_pos_array, parent_pos_array);
  std::swap(_children_pos_array, children_pos_array);
  std::swap(_load_array, load_array);
  std::swap(_nload_array, nload_array);
  std::swap(_delay_array, delay_array);
  std::swap(_ndelay_array, ndelay_array);
  std::swap(_ures_array, ures_array);
  std::swap(_ldelay_array, ldelay_array);
  std::swap(_beta_array, beta_array);
  std::swap(_impulse_array, impulse_array);
}

void RCNetCommonInfo::set_spef_cap_unit(const std::string& spef_cap_unit) {
  // The unit is 1.0 FF, fix me
  if (Str::contain(spef_cap_unit.c_str(), "1 FF") ||
      Str::contain(spef_cap_unit.c_str(), "1.0 FF")) {
    _spef_cap_unit = CapacitiveUnit::kFF;
  } else {
    _spef_cap_unit = CapacitiveUnit::kPF;
  }
}

void RCNetCommonInfo::set_spef_resistance_unit(
    const std::string& spef_resistance_unit) {
  // The unit is 1.0 OHM, fix me
  if (Str::contain(spef_resistance_unit.c_str(), "1 OHM") ||
      Str::contain(spef_resistance_unit.c_str(), "1.0 OHM")) {
    _spef_resistance_unit = ResistanceUnit::kOHM;
  } else {
    _spef_resistance_unit = ResistanceUnit::kOHM;
  }
}

std::string RcNet::name() const { return _net->get_name(); }
size_t RcNet::numPins() const { return _net->get_pin_ports().size() - 1; }

/**
 * @brief construct each net as a rctree
 *
 * @param  spef_net
 * @return rctree of a net.
 * steps:1、create a rctree 2、insert the node and their capacitance
 * 3、insert the segment(sub net)and resistance
 */
void RcNet::makeRct(RustSpefNet* rust_spef_net) {
  auto& rct = _rct.emplace<RcTree>();

  static auto* rc_net_common_info = RcNet::get_rc_net_common_info();
  static auto spef_cap_unit = rc_net_common_info->get_spef_cap_unit();
  static auto uniform_cap_unit = CapacitiveUnit::kPF;

  {
    void* spef_net_conn;
    FOREACH_VEC_ELEM(&(rust_spef_net->_conns), void, spef_net_conn) {
      auto* rust_spef_conn = static_cast<RustSpefConnEntry*>(
          rust_convert_spef_conn(spef_net_conn));

      rct.insertNode(rust_spef_conn->_name,
                     ConvertCapUnit(spef_cap_unit, uniform_cap_unit,
                                    rust_spef_conn->_load));
      rust_free_spef_conn(rust_spef_conn);
    }
  }

  {
    void* spef_net_cap;
    FOREACH_VEC_ELEM(&(rust_spef_net->_caps), void, spef_net_cap) {
      auto* rust_spef_cap = static_cast<RustSpefResCap*>(
          rust_convert_spef_net_cap_res(spef_net_cap));

      // Ground cap, otherwise couple cap
      std::string node1 = rust_spef_cap->_node1;
      std::string node2 = rust_spef_cap->_node2;
      if (node2.empty()) {
        rct.insertNode(node1, ConvertCapUnit(spef_cap_unit, uniform_cap_unit,
                                             rust_spef_cap->_res_or_cap));
      } else {
        rct.insertNode(node1, node2, rust_spef_cap->_res_or_cap);
      }

      rust_free_spef_net_cap_res(rust_spef_cap);
    }
  }

  {
    void* spef_net_res;
    FOREACH_VEC_ELEM(&(rust_spef_net->_ress), void, spef_net_res) {
      auto* rust_spef_res = static_cast<RustSpefResCap*>(
          rust_convert_spef_net_cap_res(spef_net_res));

      std::string node1 = rust_spef_res->_node1;
      std::string node2 = rust_spef_res->_node2;

      rct.insertSegment(node1, node2, rust_spef_res->_res_or_cap);

      rust_free_spef_net_cap_res(rust_spef_res);
    }
  }
}

/**
 * @brief Tranverse from the node by dfs.
 *
 * @param parent
 * @param node
 */
void RcNet::dfsTranverse(RctNode* parent, RctNode& node) {
  if (node.isVisited()) {
    return;
  }

  if (node.isTranverse()) {
    _is_found_loop = true;
    _rc_loop.push(&node);
    node.set_is_tranverse(false);
    return;
  }

  node.set_is_tranverse(true);
  bool is_all_visited = true;
  for (auto fanout_edge : node._fanout) {
    if (fanout_edge->isVisited()) {
      continue;
    } else {
      is_all_visited = false;
    }

    auto& to_node = fanout_edge->_to;

    if (fanout_edge->isBreak() || &to_node == parent) {
      continue;
    }

    fanout_edge->set_is_visited(true);

    dfsTranverse(&node, to_node);
    if (_is_found_loop) {
      _rc_loop.push(&node);
      is_all_visited = false;
      break;
    }
  }

  if (is_all_visited) {
    node.set_is_visited(true);
  }

  node.set_is_tranverse(false);
}

/**
 * @brief Find the rc tree loop.
 *
 */
void RcNet::checkLoop() {
  auto& rct = std::get<RcTree>(_rct);
  auto& nodes = rct.get_nodes();
  _is_found_loop = false;

  while (true) {
    bool need_check_again = false;
    for (auto& [node_name, node] : nodes) {
      // std::cout << "check node " << ++i << " " << node_name << std::endl;
      dfsTranverse(nullptr, node);
      if (_is_found_loop) {
        breakLoop();
        _is_found_loop = false;
        need_check_again = true;
      }
    }

    if (!need_check_again) {
      break;
    }

    for (auto& [node_name, node] : nodes) {
      node.set_is_visited(false);
      node.set_is_tranverse(false);
    }

    for (auto& edge : rct.get_edges()) {
      edge.set_is_visited(false);
    }
  }

  std::vector<RctEdge*> to_be_deleted_edge;
  for (auto& edge : rct.get_edges()) {
    if (edge.isBreak()) {
      to_be_deleted_edge.push_back(&edge);
    }
  }

  for (auto* del_edge : to_be_deleted_edge) {
    del_edge->get_from().removeFanout(del_edge);
    del_edge->get_to().removeFanin(del_edge);
    rct.removeEdge(del_edge);
  }

  rct.resetNodeVisit();
}

/**
 * @brief Break the edge loop.
 *
 */
void RcNet::breakLoop() {
  auto& rct = std::get<RcTree>(_rct);
  auto cmp = [](RctEdge* left, RctEdge* right) {
    return left->get_res() > right->get_res();
  };
  std::priority_queue<RctEdge*, std::vector<RctEdge*>, decltype(cmp)>
      loop_edges(cmp);

  bool find_loop_edge = true;
  auto* the_node = _rc_loop.front();
  auto* loop_node = the_node;
  _rc_loop.pop();

  while (!_rc_loop.empty()) {
    auto* next_node = _rc_loop.front();

    auto add_loop_edge = [&loop_edges, &rct](auto& from, auto& to) {
      auto edge = rct.findEdge(from, to);
      LOG_FATAL_IF(!edge);
      loop_edges.push(*edge);
    };

    if (find_loop_edge) {
      add_loop_edge(*the_node, *next_node);
      add_loop_edge(*next_node, *the_node);
    }

    // find all edge of the loop, stop find edge.
    if (next_node == loop_node) {
      find_loop_edge = false;
    }

    next_node->set_is_tranverse(false);
    the_node = next_node;

    _rc_loop.pop();
  }

  auto* the_least_res_edge = loop_edges.top();
  the_least_res_edge->set_is_break();

  auto another_res_edge = rct.findEdge(the_least_res_edge->get_to(),
                                       the_least_res_edge->get_from());
  LOG_FATAL_IF(!another_res_edge);
  (*another_res_edge)->set_is_break();
}

/**
 * @brief updateTiming
 *
 * @param parser of .spef
 * @return upadate the delay of each rctree node
 * steps: 1、construce rctree 2、determine the root of rctree 3.update timing
 */
void RcNet::updateRcTreeInfo() {
  auto* driver = _net->getDriver();
  auto pin_ports = _net->get_pin_ports();

  // fix for net is only driver
  if (pin_ports.size() < 2 || driver == nullptr) {
    return;
  }

  if (_rct.index() == 0) {
    std::get<EmptyRct>(_rct).load =
        std::accumulate(pin_ports.begin(), pin_ports.end(), 0.0,
                        [this, driver](double v, DesignObject* pin) {
                          return pin == driver ? v : v + pin->cap();
                        });
  } else {
    auto& rct = std::get<RcTree>(_rct);
    for (auto* pin : pin_ports) {
      if (auto* node = rct.rcNode(pin->getFullName()); node) {
        if (pin == driver) {
          rct._root = node;
          node->set_is_root();
        }
        node->set_obj(pin);
      } else {
        const auto& nodes = rct.get_nodes();
        for (const auto& [node_name, node] : nodes) {
          LOG_INFO << node_name;
        }

        LOG_FATAL << "pin " << pin->getFullName() << " can not found in RCTree "
                  << name() << std::endl;
      }
    }
    // if (name() == "fanout_buf_215") {
    //   rct.printGraphViz();
    // }
    LOG_FATAL_IF(!rct._root)
        << "not found rct root for net " << _net->get_name();
  }

  // printRctInfo();
}

/**
 * @brief updateTiming
 *
 * @param parser of .spef
 * @return upadate the delay of each rctree node
 * steps: 1、construce rctree 2、determine the root of rctree 3.update timing
 */
void RcNet::updateRcTiming(RustSpefNet* spef_net) {
  makeRct(spef_net);
  updateRcTreeInfo();

  //  not empty Rct.
  if (_rct.index() != 0) {
    checkLoop();

    auto& rct = std::get<RcTree>(_rct);
    rct.updateRcTiming();

    // the previous is annotated.(for test.)
    // nangate45:"FE_OFN0_text_out_80"
    // if (name() == "in1" || name() == "r2q" || name() == "u1z" ||
    //     name() == "u2z" || name() == "out") {
    //   rct.printGraphViz();
    // }
  }
}
/**
 * @brief net load
 *
 * @param
 * @return the total load of a net
 * The total capacitive load is defined as the sum of the input capacitance
 * of all the other devices sharing the trace.
 */

double RcNet::load() {
  if (_rct.index() == 0) {
    return std::get<EmptyRct>(_rct).load;
  } else {
    return std::get<RcTree>(_rct)._root->_load;
  }
}

/**
 * @brief net load
 *
 * @param
 * @return the total load of a net
 * The total capacitive load is defined as the sum of the input capacitance
 * of all the other devices sharing the trace.
 */
double RcNet::load(AnalysisMode mode, TransType trans_type) {
  if (_rct.index() == 0) {
    return std::get<EmptyRct>(_rct).load;
  } else {
    if (std::get<RcTree>(_rct)._root) {
      return std::get<RcTree>(_rct)
          ._root->_nload[ModeTransPair(mode, trans_type)];
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief get slew impulse for gpu speedup data.
 *
 * @param mode
 * @param trans_type
 * @return double
 */
double RcNet::slewImpulse(DesignObject& to, AnalysisMode mode,
                          TransType trans_type) {
  if (!rct()) {
    return 0.0;
  }

  auto* node = std::get<RcTree>(_rct).node(to.getFullName());
  if (_rct.index() == 0) {
    return 0.0;
  } else {
    if (node) {
      return node->_impulse[ModeTransPair{mode, trans_type}];
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Get the net load nodes.
 *
 * @return std::vector<RctNode*>
 */
std::set<RctNode*> RcNet::getLoadNodes() {
  RcTree& rc_tree = std::get<RcTree>(_rct);
  std::vector<DesignObject*> lnodes = _net->getLoads();
  std::set<RctNode*> load_nodes;
  for (auto* lnode : lnodes) {
    std::string lname = lnode->getFullName();
    RctNode* rcnode = rc_tree.node(lname);
    load_nodes.insert(rcnode);
  }

  return load_nodes;
}

/**
 * @brief Get node load.
 *
 * @param node_name
 * @return double
 */
double RcNet::getNodeLoad(const char* node_name) {
  double load = 0.0;

  if (auto* rc_tree = rct(); rc_tree) {
    auto* node = rc_tree->node(node_name);
    load = node->_load;
  }

  return load;
}

/**
 * @brief Get rc net resistance from driver pin to load pin.
 *
 * @param mode
 * @param trans_type
 * @param load_obj
 * @return double
 */
double RcNet::getResistance(AnalysisMode mode, TransType trans_type,
                            DesignObject* load_obj) {
  double res = 0.0;

  if (auto* rc_tree = rct(); rc_tree) {
    std::string node_name = load_obj->getFullName();
    auto* node = rc_tree->node(node_name);

    res = node->get_ures(mode, trans_type);
  }

  return res;
}

/**
 * @brief Get rc net resistance from driver pin to load pin.
 *
 * @param node_name
 * @return double
 */
double RcNet::getNodeResistance(const char* node_name) {
  double res = 0.0;

  if (auto* rc_tree = rct(); rc_tree) {
    auto* node = rc_tree->node(node_name);
    res = node->_res;
  }

  return res;
}

/**
 * @brief Get rc net resistance.
 *
 * @return double
 */
double RcNet::getNetResistance() {
  double res = 0.0;

  if (auto* rc_tree = rct(); rc_tree) {
    for (auto& edge : rc_tree->_edges) {
      res += edge.get_res();
    }
  }

  return res;
}

/**
 * @brief get delay of rc node.
 *
 * @param to
 * @param delay_method
 * @return std::optional<double>
 */
std::optional<double> RcNet::delay(const char* node_name,
                                   DelayMethod delay_method) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto node = std::get<RcTree>(_rct).node(node_name);
  std::optional<double> delay;
  if (delay_method == DelayMethod::kElmore) {
    delay = node->delay();
  } else if (delay_method == DelayMethod::kD2M) {
    delay = node->delayD2M();
  } else if (delay_method == DelayMethod::kECM) {
    delay = node->delayECM();
  } else {
    delay = node->delayD2MM();
  }
  return delay;
}

/**
 * @brief get delay of pin or port.
 *
 * @param to
 * @param delay_method
 * @return std::optional<double>
 */
std::optional<double> RcNet::delay(DesignObject& to, DelayMethod delay_method) {
  return delay(to.getFullName().c_str());
}

std::optional<std::pair<double, Eigen::MatrixXd>> RcNet::delay(
    DesignObject& to, double /* from_slew */,
    std::optional<LibCurrentData*> /* output_current */, AnalysisMode mode,
    TransType trans_type) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto* node = std::get<RcTree>(_rct).node(to.getFullName());
  Eigen::MatrixXd waveform;
  return std::make_pair(node->delay(mode, trans_type), waveform);
}

std::optional<double> RcNet::slew(const char* node_name, double from_slew,
                                  AnalysisMode mode, TransType trans_type) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto* node = std::get<RcTree>(_rct).node(node_name);
  double slew = node->slew(mode, trans_type, from_slew);

  return slew;
}

std::optional<double> RcNet::slew(
    DesignObject& to, double from_slew,
    std::optional<LibCurrentData*> /* output_current */, AnalysisMode mode,
    TransType trans_type) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto* node = std::get<RcTree>(_rct).node(to.getFullName());
  if (!node) {
    printRctInfo();
    LOG_FATAL << "node " << to.getFullName() << " not found in the RC Tree.";
  }

  return node->slew(mode, trans_type, from_slew);
}

/**
 * @brief Get all node slew based on the from driver slew.
 *
 * @param driver_slew
 * @param mode
 * @param trans_type
 * @return std::map<std::string, double>
 */
std::map<std::string, double>& RcNet::getAllNodeSlew(double driver_slew,
                                                    AnalysisMode mode,
                                                    TransType trans_type) {
  if (_all_node_slews) {
    return *_all_node_slews;
  }

  _all_node_slews = std::map<std::string, double>{};
  auto* rc_tree = rct();
  if (!rc_tree) {
    return _all_node_slews.value();
  }

  auto* rc_root = rc_tree->_root;
  if (!rc_root) {
    return _all_node_slews.value();
  }

  _all_node_slews.value()[rc_root->get_name()] = driver_slew;
  
  std::map<std::string, double> all_node_slews;

  std::function<void(RctNode*, RctNode*)> get_snk_slew =
      [&get_snk_slew, this, mode, trans_type, driver_slew](
          RctNode* parent_node, RctNode* src_node) {        
        auto& fanout_edges = src_node->get_fanout();
        for (auto* fanout_edge : fanout_edges) {          
          auto& snk_node = fanout_edge->_to;
          if (fanout_edge->isBreak() || &snk_node == parent_node) {
            continue;
          }

          auto snk_slew = snk_node.slew(mode, trans_type, NS_TO_PS(driver_slew));
          _all_node_slews.value()[snk_node.get_name()] = PS_TO_NS(snk_slew);

          get_snk_slew(src_node, &snk_node);
        }
      };

  get_snk_slew(nullptr, rc_root);

  return _all_node_slews.value();
}

/**
 * @brief From the driver node, get the edges to the required node.
 *
 * @param to_node_name
 * @return std::vector<RctEdge*>
 */
std::vector<RctEdge*> RcNet::getWireTopo(const char* to_node_name) {
  std::vector<RctEdge*> wire_topo;
  if (_rct.index() == 0) {
    return wire_topo;
  }

  std::function<unsigned(RctNode*, RctNode*)> get_topo_edge =
      [&get_topo_edge, this, &wire_topo, to_node_name](
          RctNode* parent_node, RctNode* src_node) -> unsigned {
    auto& fanout_edges = src_node->get_fanout();
    for (auto* fanout_edge : fanout_edges) {
      auto& snk_node = fanout_edge->_to;
      if (fanout_edge->isBreak() || &snk_node == parent_node) {
        continue;
      }

      if (snk_node.get_name() == to_node_name) {
        wire_topo.push_back(fanout_edge);
        return 1;
      }

      if (get_topo_edge(src_node, &snk_node)) {
        wire_topo.push_back(fanout_edge);
        return 1;
      }
    }

    return 0;
  };

  auto* rc_tree = rct();
  get_topo_edge(nullptr, rc_tree->_root);

  LOG_FATAL_IF((wire_topo.empty() ||
               (*wire_topo.begin())->_to.get_name() != to_node_name))
      << "not found to node name " << to_node_name;

  return wire_topo;
}

/**
 * @brief Print RC Tree info.
 *
 */
void RcNet::printRctInfo() {
  auto& nodes = std::get<RcTree>(_rct)._str2nodes;
  DLOG_INFO << "node num: " << nodes.size() << "\n";

  for (auto& tnode : nodes) {
    DLOG_INFO << tnode.second._name << "\n";
    DLOG_INFO << "load:" << tnode.second._load << std::endl;
    DLOG_INFO << "cap:" << tnode.second.cap() << std::endl;
    DLOG_INFO << "delay:" << tnode.second._delay << std::endl;
  }
}
}  // namespace ista