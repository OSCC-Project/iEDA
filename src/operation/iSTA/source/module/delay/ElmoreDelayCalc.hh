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
 * @file DelayCalc.h
 * @author LH (liuh0326@163.com)
 * @brief The class of elmore delay calc method.
 * @version 0.1
 * @date 2021-01-27
 */

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <variant>

#include "WaveformApproximation.hh"
#include "liberty/Liberty.hh"
#include "netlist/Net.hh"
#include "netlist/Pin.hh"
#include "netlist/Port.hh"
#include "spef/parser-spef.hpp"

namespace ista {
class RctEdge;
class RctNode;
class RcTree;
class LibetyCurrentData;

/**
 * @brief The RC tree node, that has ground capacitance.
 *
 */
class RctNode {
  friend class RcTree;
  friend class RcNet;
  friend class ArnoldiNet;

 public:
  RctNode() = default;
  explicit RctNode(std::string&&);

  virtual ~RctNode() = default;

  void set_is_root() { _is_root = 1; }
  [[nodiscard]] unsigned isRoot() const { return _is_root; }

  [[nodiscard]] double nodeLoad() const { return _load; }
  double nodeLoad(AnalysisMode mode, TransType trans_type);
  [[nodiscard]] double cap() const { return _obj ? _obj->cap() + _cap : _cap; }
  [[nodiscard]] double get_cap() const { return _cap; }
  double cap(AnalysisMode mode, TransType trans_type);
  double get_cap(AnalysisMode mode, TransType trans_type) {
    return _ncap[ModeTransPair(mode, trans_type)];
  }
  void setCap(double cap);
  void incrCap(double cap);

  double get_ures(AnalysisMode mode, TransType trans_type) {
    return _ures[ModeTransPair(mode, trans_type)];
  }

  void calNodePIModel();
  void set_pi(PiModel* pi) {
    _pi.C_far = pi->C_far;
    _pi.C_near = pi->C_near;
    _pi.R = pi->R;
  }

  [[nodiscard]] double delay() const { return _delay; }
  double delay(AnalysisMode mode, TransType trans_type);
  [[nodiscard]] double delayD2M() const {
    return _m2 == 0 ? 0 : _delay * _delay / sqrt(_m2) * log(2);
  }
  [[nodiscard]] double delayECM() const { return _delay_ecm; }
  [[nodiscard]] double delayD2MM() const {
    return _m2_c == 0 ? 0 : _delay_ecm * _delay_ecm / sqrt(_m2_c) * log(2);
  }
  double updateCeff() {
    calNodePIModel();
    if (_moments.y2 == 0 && _moments.y3 == 0) {
      _ceff = nodeLoad();
    } else {
      _ceff = _pi.C_near + _pi.C_far * (1 - exp(-_delay / (_pi.R * _pi.C_far)));
    }
    return _ceff;
  }
  double slew(AnalysisMode mode, TransType trans_type, double input_slew);
  [[nodiscard]] std::string get_name() const { return _name; }
  void set_cap(double cap) { _cap = cap; }
  [[nodiscard]] unsigned isUpdateLoad() const { return _is_update_load; }
  void set_is_update_load(bool updated) { _is_update_load = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateDelay() const { return _is_update_delay; }
  void set_is_update_delay(bool updated) {
    _is_update_delay = (updated ? 1 : 0);
  }

  [[nodiscard]] unsigned isUpdateDelayECM() const {
    return _is_update_delay_ecm;
  }
  void set_is_update_delay_ecm(bool updated) {
    _is_update_delay_ecm = (updated ? 1 : 0);
  }

  [[nodiscard]] unsigned isUpdateM2() const { return _is_update_m2; }
  void set_is_update_m2(bool updated) { _is_update_m2 = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateMC() const { return _is_update_mc; }
  void set_is_update_mc(bool updated) { _is_update_mc = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateM2C() const { return _is_update_m2_c; }
  void set_is_update_m2_c(bool updated) { _is_update_m2_c = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateMCC() const { return _is_update_mc_c; }
  void set_is_update_mc_c(bool updated) { _is_update_mc_c = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateCeff() const { return _is_update_ceff; }
  void set_is_update_ceff(bool updated) { _is_update_ceff = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isUpdateLdelay() const { return _is_update_ldelay; }
  void set_is_update_Ldelay(bool updated) {
    _is_update_ldelay = (updated ? 1 : 0);
  }

  [[nodiscard]] unsigned isUpdateResponse() const {
    return _is_update_response;
  }
  void set_is_update_response(bool updated) {
    _is_update_response = (updated ? 1 : 0);
  }

  [[nodiscard]] unsigned isTranverse() const { return _is_tranverse; }
  void set_is_tranverse(bool updated) { _is_tranverse = (updated ? 1 : 0); }

  [[nodiscard]] unsigned isVisited() const { return _is_visited; }
  void set_is_visited(bool updated) { _is_visited = (updated ? 1 : 0); }

  void set_is_visited_ecm(bool updated) { _is_visited_ecm = (updated ? 1 : 0); }
  [[nodiscard]] unsigned isVisitedEcm() const { return _is_visited_ecm; }

  void set_obj(DesignObject* obj) { _obj = obj; }
  DesignObject* get_obj() { return _obj; }

  LaplaceMoments* get_moments() { return &_moments; }

  auto& get_fanin() { return _fanin; }
  auto& get_fanout() { return _fanout; }

  void removeFanin(RctEdge* the_edge) {
    auto it = std::find_if(_fanin.begin(), _fanin.end(),
                           [the_edge](auto* edge) { return the_edge == edge; });
    LOG_FATAL_IF(it == _fanin.end());
    _fanin.erase(it);
  }

  void removeFanout(RctEdge* the_edge) {
    auto it = std::find_if(_fanout.begin(), _fanout.end(),
                           [the_edge](auto* edge) { return the_edge == edge; });
    LOG_FATAL_IF(it == _fanout.end());
    _fanout.erase(it);
  }

 private:
  std::string _name;

  double _cap = 0.0;
  double _load = 0.0;
  double _delay = 0.0;
  double _mc = 0.0;    //!< Elmore * cap
  double _m2 = 0.0;    //!< The two moment.
  double _mc_c = 0.0;  //!< Elmore * ceff
  double _m2_c = 0.0;  //!< The two moment with ceff.
  double _ceff = 0.0;
  double _delay_ecm = 0.0;

  unsigned _is_update_load : 1 = 0;
  unsigned _is_update_delay : 1 = 0;
  unsigned _is_update_ldelay : 1 = 0;
  unsigned _is_update_delay_ecm : 1 = 0;
  unsigned _is_update_m2 : 1 = 0;
  unsigned _is_update_mc : 1 = 0;
  unsigned _is_update_m2_c : 1 = 0;
  unsigned _is_update_mc_c : 1 = 0;
  unsigned _is_update_ceff : 1 = 0;
  unsigned _is_update_response : 1 = 0;
  unsigned _is_tranverse : 1 = 0;
  unsigned _is_visited : 1 = 0;
  unsigned _is_visited_ecm : 1 = 0;
  unsigned _is_root : 1 = 0;
  unsigned _reserved : 18 = 0;

  std::map<ModeTransPair, double> _ures;
  std::map<ModeTransPair, double> _nload;
  std::map<ModeTransPair, double> _beta;
  std::map<ModeTransPair, double> _ncap;
  std::map<ModeTransPair, double> _ndelay;
  std::map<ModeTransPair, double> _ldelay;
  std::map<ModeTransPair, double> _impulse;

  std::list<RctEdge*> _fanin;
  std::list<RctEdge*> _fanout;

  DesignObject* _obj{nullptr};

  LaplaceMoments _moments;
  PiModel _pi;

  FORBIDDEN_COPY(RctNode);
};

/**
 * @brief Traverse fanout edge of the node, usage:
 * FOREACH_RCNODE_FANOUT_EDGE(node, edge)
 * {
 *    do_something_for_edge();
 * }
 */
#define FOREACH_RCNODE_FANOUT_EDGE(node, edge) \
  for (auto* edge : (node)->get_fanout())

/**
 * @brief Traverse fanin edge of the node, usage:
 * FOREACH_RCNODE_FANIN_EDGE(node, edge)
 * {
 *    do_something_for_edge();
 * }
 */
#define FOREACH_RCNODE_FANIN_EDGE(node, edge) \
  for (auto* edge : (node)->get_fanin())

/**
 * @brief The class for the coupled rc node.
 *
 */
class CoupledRcNode {
 public:
  CoupledRcNode(const std::string& aggressor_node,
                const std::string& victim_node, double coupled_cap)
      : _local_node(aggressor_node),
        _remote_node(victim_node),
        _coupled_cap(coupled_cap) {}
  ~CoupledRcNode() = default;

  std::string& get_local_node() { return _local_node; }
  std::string& get_remote_node() { return _remote_node; }
  double get_coupled_cap() const { return _coupled_cap; }

 private:
  std::string _local_node;   //!< for spef coupling capacitor, the first node is
                             //!< local node.
  std::string _remote_node;  // the second node is remote node.
  double _coupled_cap;
};

/**
 * @brief The RC tree edge, that has resistance.
 *
 */
class RctEdge {
  friend class RcTree;
  friend class RcNet;
  friend class ArnoldiNet;

 public:
  RctEdge(RctNode&, RctNode&, double);
  ~RctEdge() = default;

  [[nodiscard]] double get_res() const { return _res; }
  void set_res(double r) { _res = r; }
  [[nodiscard]] double getG(AnalysisMode /* mode */,
                            TransType /* trans_type */) const {
    return 1 / _res;
  }

  RctNode& get_from() { return _from; }
  RctNode& get_to() { return _to; }
  void set_is_in_order(bool is_in_order) { _is_in_order = is_in_order; }
  [[nodiscard]] bool isInOrder() const { return _is_in_order; }
  void set_is_break() { _is_break = true; }
  [[nodiscard]] bool isBreak() const { return _is_break; }

  void set_is_visited(bool is_visited) { _is_visited = is_visited; }
  [[nodiscard]] bool isVisited() const { return _is_visited; }

  friend bool operator==(const RctEdge& lhs, const RctEdge& rhs) {
    return (&lhs == &rhs);
  }

 private:
  RctNode& _from;
  RctNode& _to;

  unsigned _is_break : 1 = 0;
  unsigned _is_visited : 1 = 0;
  unsigned _is_in_order : 1 = 0;
  unsigned _reserved : 30 = 0;

  double _res = 0.0;

  FORBIDDEN_COPY(RctEdge);
};

/**
 * @brief The RC tree, consist of resistance, ground capacitance.
 *
 */
class RcTree {
  friend class RcNet;
  friend class ArnoldiNet;

  friend void swap(RcTree& lhs, RcTree& rhs) {
    std::swap(lhs._root, rhs._root);
    std::swap(lhs._str2nodes, rhs._str2nodes);
    std::swap(lhs._edges, rhs._edges);
  }

 public:
  RcTree() = default;
  ~RcTree() = default;

  void updateRcTiming();
  void insertSegment(const std::string&, const std::string&, double);
  RctNode* insertNode(const std::string&, double = 0.0);
  CoupledRcNode* insertNode(const std::string& local_node,
                            const std::string& remote_node, double coupled_cap);
  RctEdge* insertEdge(const std::string&, const std::string&, double);
  RctEdge* insertEdge(RctNode* node1, RctNode* node2, double res,
                      bool in_order);

  double delay(const std::string& name);
  double delay(const std::string& name, AnalysisMode mode,
               TransType trans_type);

  double slew(const std::string& name, AnalysisMode mode, TransType trans_type,
              double input_slew);

  [[nodiscard]] size_t numNodes() const { return _str2nodes.size(); }
  [[nodiscard]] size_t numEdges() const { return _edges.size(); }

  auto* get_root() { return _root; }
  void set_root(RctNode* root_node) { _root = root_node; }
  auto& get_nodes() { return _str2nodes; }
  auto get_node_num() { return _str2nodes.size(); }
  auto& get_edges() { return _edges; }
  auto& get_coupled_nodes() { return _coupled_nodes; }

  void removeEdge(RctEdge* the_edge) {
    auto it =
        std::find_if(_edges.begin(), _edges.end(),
                     [the_edge](auto& edge) { return the_edge == &edge; });
    LOG_FATAL_IF(it == _edges.end());
    _edges.erase(it);
  }

  void removeNode(RctNode* the_node) {
    for (auto* fanin_edge : the_node->get_fanin()) {
      fanin_edge->get_from().removeFanout(fanin_edge);
    }

    for (auto* fanout_edge : the_node->get_fanout()) {
      fanout_edge->get_to().removeFanin(fanout_edge);
    }

    auto it =
        std::find_if(_str2nodes.begin(), _str2nodes.end(),
                     [the_node](auto& it) { return &(it.second) == the_node; });
    LOG_FATAL_IF(it == _str2nodes.end());
    _str2nodes.erase(it);

    auto it1 =
        std::find_if(_edges.begin(), _edges.end(), [the_node](auto& edge) {
          return &(edge.get_from()) == the_node || &(edge.get_to()) == the_node;
        });
    LOG_FATAL_IF(it1 == _edges.end());
    _edges.erase(it1);
  }

  std::optional<RctEdge*> findEdge(RctNode& from, RctNode& to) {
    auto it = std::find_if(_edges.begin(), _edges.end(), [&](RctEdge& edge) {
      return &from == &edge.get_from() && &to == &edge.get_to();
    });
    if (it != _edges.end()) {
      return &(*it);
    }
    return std::nullopt;
  }

  std::optional<RctEdge*> findEdge(std::string from_name, std::string to_name) {
    auto it = std::find_if(_edges.begin(), _edges.end(), [&](RctEdge& edge) {
      return from_name == edge.get_from().get_name() &&
             to_name == edge.get_to().get_name();
    });
    if (it != _edges.end()) {
      return &(*it);
    }
    return std::nullopt;
  }

  RctNode* node(const std::string&);

  void resetNodeVisit() {
    for (auto& node : _str2nodes) {
      node.second.set_is_visited(false);
    }
  }

  bool isHaveCoupledNodes() { return !_coupled_nodes.empty(); }

  void printGraphViz();

 private:
  RctNode* _root{nullptr};

  std::map<std::string, RctNode> _str2nodes;
  std::list<RctEdge> _edges;

  std::vector<CoupledRcNode> _coupled_nodes;

  void initData();
  void initMoment();
  void updateLoad(RctNode*, RctNode*);
  void updateMC(RctNode* parent, RctNode* from);
  void updateMCC(RctNode* parent, RctNode* from);
  void updateDelay(RctNode*, RctNode*);
  void updateDelayECM(RctNode* parent, RctNode* from);
  void updateM2(RctNode* parent, RctNode* from);
  void updateM2C(RctNode* parent, RctNode* from);
  void updateLDelay(RctNode* parent, RctNode* from);
  void updateResponse(RctNode* parent, RctNode* from);

  RctNode* rcNode(const std::string&);

  FORBIDDEN_COPY(RcTree);
};

/**
 * @brief Traverse node of the tree, usage:
 * FOREACH_RCTREE_NODE(tree, node)
 * {
 *    do_something_for_node();
 * }
 */
#define FOREACH_RCTREE_NODE(tree, name, node) \
  for (auto& [name, node] : tree.get_nodes())

/**
 * @brief Traverse edge of the tree, usage:
 * FOREACH_RCTREE_EDGE(tree, edge)
 * {
 *    do_something_for_edge();
 * }
 */
#define FOREACH_RCTREE_EDGE(tree, edge) for (auto& edge : tree.get_edges())

/**
 * @brief The spef common head information.
 *
 */
class RCNetCommonInfo {
 public:
  void set_spef_cap_unit(const std::string& spef_cap_unit) {
    // The unit is 1.0 FF, fix me
    if (Str::contain(spef_cap_unit.c_str(), "FF")) {
      _spef_cap_unit = CapacitiveUnit::kFF;
    } else {
      _spef_cap_unit = CapacitiveUnit::kPF;
    }
  }
  void set_spef_resistance_unit(const std::string& spef_resistance_unit) {
    // The unit is 1.0 OHM, fix me
    if (Str::contain(spef_resistance_unit.c_str(), "OHM")) {
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
 * @brief The RC net, that is the top elmore calc interface of the net.
 *
 */
class RcNet {
 public:
  struct EmptyRct {
    double load;
  };
  enum class DelayMethod { kElmore, kD2M, kECM, kD2MC };

  explicit RcNet(Net* net) : _net(net) {}
  virtual ~RcNet() = default;

  [[nodiscard]] std::string name() const;
  [[nodiscard]] size_t numPins() const;

  [[nodiscard]] auto* get_net() const { return _net; }

  RcTree* rct() { return std::get_if<RcTree>(&_rct); }
  void makeRct() { _rct.emplace<RcTree>(); }
  virtual void makeRct(const spef::Net& spef_net);
  void updateRcTreeInfo();
  virtual void dfsTranverse(RctNode* parent, RctNode& node);
  virtual void checkLoop();
  virtual void breakLoop();
  virtual void updateRcTiming(const spef::Net& spef_net);

  double load();
  double load(AnalysisMode mode, TransType trans_type);
  std::set<RctNode*> getLoadNodes();

  double getResistance(AnalysisMode mode, TransType trans_type,
                       DesignObject* load_obj);

  std::optional<double> delay(DesignObject& to,
                              DelayMethod delay_method = DelayMethod::kElmore);
  std::optional<double> delayNs(DesignObject& to, DelayMethod delay_method) {
    auto delay_ps = delay(to, delay_method);
    if (delay_ps) {
      return PS_TO_NS(delay_ps.value());
    }
    return std::nullopt;
  }

  virtual std::optional<std::pair<double, Eigen::MatrixXd>> delay(
      DesignObject& to, double from_slew,
      std::optional<LibetyCurrentData*> output_current, AnalysisMode mode,
      TransType trans_type);

  virtual std::optional<double> slew(
      Pin& to, double from_slew,
      std::optional<LibetyCurrentData*> output_current, AnalysisMode mode,
      TransType trans_type);

  void printRctInfo();

  static void set_rc_net_common_info(
      std::unique_ptr<RCNetCommonInfo>&& rc_net_common_info) {
    _rc_net_common_info = std::move(rc_net_common_info);
  }

  static RCNetCommonInfo* get_rc_net_common_info() {
    return _rc_net_common_info.get();
  }

 protected:
  Net* _net;
  std::variant<EmptyRct, RcTree> _rct;

  std::queue<RctNode*> _rc_loop;
  bool _is_found_loop = false;

 private:
  static std::unique_ptr<RCNetCommonInfo> _rc_net_common_info;
};

}  // namespace ista
