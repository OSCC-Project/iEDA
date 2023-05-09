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
 * @file DelayCalc.cc
 * @author LH (liuh0326@163.com)
 * @brief The file implement traditional elmore calc algorithm.
 * @version 0.1
 * @date 2021-01-27
 */

#include "ElmoreDelayCalc.hh"

#include <numeric>
#include <queue>
#include <utility>

#include "log/Log.hh"
#include "spef/parser-spef.hpp"

namespace ista {

std::unique_ptr<RCNetCommonInfo> RcNet::_rc_net_common_info;

RctNode::RctNode(std::string&& name)
    : _name{std::move(name)},
      _is_update_load(0),
      _is_update_delay(0),
      _is_update_ldelay(0),
      _is_update_response(0),
      _is_tranverse(0),
      _is_visited(0) {}

double RctNode::nodeLoad(AnalysisMode mode, TransType trans_type) {
  return _nload[ModeTransPair(mode, trans_type)];
}

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
    : _from{from}, _to{to}, _res{res} {
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
  if (_str2nodes.contains(name) && IsDoubleEqual(_str2nodes[name]._cap, cap)) {
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
 * @brief calculate and update the load of each node,calculate and update the
 * delay from net root to each node.
 */
void RcTree::updateRcTiming() {
  if (!_root) {
    LOG_ERROR << "RCTree root can not found";
    return;
  }

  initData();

  updateLoad(nullptr, _root);
  updateDelay(nullptr, _root);
  updateLDelay(nullptr, _root);
  updateResponse(nullptr, _root);

  // printGraphViz();
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
 * @brief upadate the delay from net root to each node
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
 * @brief Print the rc tree to graphviz dot file format.
 *
 */
void RcTree::printGraphViz() {
  LOG_INFO << "dump graph dotviz start";

  std::ofstream dot_file;
  dot_file.open("./tree.dot");

  dot_file << "digraph tree {\n";

  for (auto& edge : _edges) {
    // if (!edge.isInOrder()) {
    //   continue;
    // }
    auto from_name = edge._from.get_name();
    auto to_name = edge._to.get_name();

    dot_file << Str::printf("p%p[label=\"%s cap %f\" ]\n", &edge._from,
                            from_name.c_str(), edge._from.cap());

    dot_file << Str::printf("p%p", &edge._from) << " -> "
             << Str::printf("p%p", &edge._to)
             << Str::printf("[label=\"res %f\" ]", edge.get_res()) << "\n";

    dot_file << Str::printf("p%p[label=\"%s cap %f\" ]\n", &edge._to,
                            to_name.c_str(), edge._to.cap());
  }

  dot_file << "}\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";
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
void RcNet::makeRct(const spef::Net& spef_net) {
  auto& rct = _rct.emplace<RcTree>();

  static auto* rc_net_common_info = RcNet::get_rc_net_common_info();
  static auto spef_cap_unit = rc_net_common_info->get_spef_cap_unit();
  static auto uniform_cap_unit = CapacitiveUnit::kPF;

  for (const auto& conn : spef_net.connections) {
    rct.insertNode(conn.name,
                   (conn.load ? ConvertCapUnit(spef_cap_unit, uniform_cap_unit,
                                               *(conn.load))
                              : 0.0));
  }

  for (const auto& [node1, node2, cap] : spef_net.caps) {
    // Ground cap, otherwise couple cap
    if (node2.empty()) {
      rct.insertNode(node1,
                     ConvertCapUnit(spef_cap_unit, uniform_cap_unit, cap));
    } else {
      rct.insertNode(node1, node2, cap);
    }
  }

  for (const auto& [node1, node2, res] : spef_net.ress) {
    LOG_FATAL_IF(node2.empty());
    rct.insertSegment(node1, node2, res);
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

  for (auto& [node_name, node] : nodes) {
    dfsTranverse(nullptr, node);
    if (_is_found_loop) {
      breakLoop();
      _is_found_loop = false;
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
void RcNet::updateRcTiming(const spef::Net& spef_net) {
  makeRct(spef_net);
  updateRcTreeInfo();

  //  not empty Rct.
  if (_rct.index() != 0) {
    auto& rct = std::get<RcTree>(_rct);
    rct.updateRcTiming();

    // if (name() == "fanout_buf_215") {
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

std::optional<double> RcNet::delay(DesignObject& to) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto node = std::get<RcTree>(_rct).node(to.getFullName());
  return node->delay();
}

std::optional<std::pair<double, Eigen::MatrixXd>> RcNet::delay(
    DesignObject& to, double /* from_slew */,
    std::optional<LibetyCurrentData*> /* output_current */, AnalysisMode mode,
    TransType trans_type) {
  if (_rct.index() == 0) {
    return std::nullopt;
  }

  auto* node = std::get<RcTree>(_rct).node(to.getFullName());
  Eigen::MatrixXd waveform;
  return std::make_pair(node->delay(mode, trans_type), waveform);
}

std::optional<double> RcNet::slew(
    Pin& to, double from_slew,
    std::optional<LibetyCurrentData*> /* output_current */, AnalysisMode mode,
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
