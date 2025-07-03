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
 * @file DelayCalc.h
 * @author LH (liuh0326@163.com)
 * @brief The class of elmore delay calc method.
 * @version 0.1
 * @date 2021-01-27
 */

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <fstream>
#include <list>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <variant>
#include <memory>

#include "Type.hh"
#include "spef/SpefParserRustC.hh"
#include "WaveformApproximation.hh"
#include "log/Log.hh"

namespace ista {
class RctEdge;
class RctNode;
class RcTree;
class LibCurrentData;
class Net;
class Pin;
class Port;
class DesignObject;

/**
 * @brief The RC tree node, that has ground capacitance.
 *
 */
class RctNode {
  friend class RcTree;
  friend class RcNet;
  friend class ArnoldiNet;

 public:
  RctNode()
      : _is_update_load(0),
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
  explicit RctNode(std::string&&);

  virtual ~RctNode() = default;

  void set_is_root() { _is_root = 1; }
  [[nodiscard]] unsigned isRoot() const { return _is_root; }

  [[nodiscard]] double nodeLoad() const { return _load; }
  void set_load(double load) { _load = load; }
  double nodeLoad(AnalysisMode mode, TransType trans_type);
  [[nodiscard]] double cap() const;
  [[nodiscard]] double get_cap() const { return _cap; }
  double cap(AnalysisMode mode, TransType trans_type);
  double get_cap(AnalysisMode mode, TransType trans_type) {
    return _ncap[ModeTransPair(mode, trans_type)];
  }
  void setCap(double cap);
  void incrCap(double cap);

  std::map<ModeTransPair, double>& get_nload() { return _nload; }
  std::map<ModeTransPair, double>& get_ndelay() { return _ndelay; }
  std::map<ModeTransPair, double>& get_ures() { return _ures; }
  std::map<ModeTransPair, double>& get_ldelay() { return _ldelay; }
  std::map<ModeTransPair, double>& get_beta() { return _beta; }
  std::map<ModeTransPair, double>& get_impulse() { return _impulse; }

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
  void set_delay(double delay) { _delay = delay; }
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

  void set_flatten_pos(std::size_t flatten_pos) { _flatten_pos = flatten_pos; }
  std::size_t get_flatten_pos() { return _flatten_pos; }

  void set_parent(RctNode* parent) { _parent = parent; }
  RctNode* get_parent() { return _parent; }

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

  unsigned _is_update_load : 1;
  unsigned _is_update_delay : 1;
  unsigned _is_update_ldelay : 1;
  unsigned _is_update_delay_ecm : 1;
  unsigned _is_update_m2 : 1;
  unsigned _is_update_mc : 1;
  unsigned _is_update_m2_c : 1;
  unsigned _is_update_mc_c : 1;
  unsigned _is_update_ceff : 1;
  unsigned _is_update_response : 1;
  unsigned _is_tranverse : 1;
  unsigned _is_visited : 1;
  unsigned _is_visited_ecm : 1;
  unsigned _is_root : 1;
  unsigned _reserved : 18;

  std::map<ModeTransPair, double> _ures;
  std::map<ModeTransPair, double> _nload;
  std::map<ModeTransPair, double> _beta;
  std::map<ModeTransPair, double> _ncap;
  std::map<ModeTransPair, double> _ndelay;
  std::map<ModeTransPair, double> _ldelay;
  std::map<ModeTransPair, double> _impulse;

  std::size_t _flatten_pos = 0;
  RctNode* _parent = nullptr;
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

  double _res = 0.0;

  unsigned _is_break : 1;
  unsigned _is_visited : 1;
  unsigned _is_in_order : 1;
  unsigned _reserved : 29;

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

  RcTree(RcTree&&) noexcept = default;
  RcTree& operator=(RcTree&&) noexcept = default;

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
  const std::vector<float>& get_cap_array() const { return _cap_array; }
  const std::vector<float>& get_ncap_array() const { return _ncap_array; }
  const std::vector<float>& get_res_array() const { return _res_array; }
  const std::vector<int>& get_parent_pos_array() const {
    return _parent_pos_array;
  }
  void set_load_array(const std::vector<float>& load_array) {
    _load_array = load_array;
  }

  void set_nload_array(const std::vector<float>& nload_array) {
    _nload_array = nload_array;
  }

  void set_delay_array(const std::vector<float>& delay_array) {
    _delay_array = delay_array;
  }

  void set_ndelay_array(const std::vector<float>& ndelay_array) {
    _ndelay_array = ndelay_array;
  }

  void set_ures_array(const std::vector<float>& ures_array) {
    _ures_array = ures_array;
  }

  void set_ldelay_array(const std::vector<float>& ldelay_array) {
    _ldelay_array = ldelay_array;
  }

  void set_beta_array(const std::vector<float>& beta_array) {
    _beta_array = beta_array;
  }

  void set_impulse_array(const std::vector<float>& impulse_array) {
    _impulse_array = impulse_array;
  }

  std::vector<float>& get_load_array() { return _load_array; }

  std::vector<float>& get_nload_array() { return _nload_array; }

  std::vector<float>& get_delay_array() { return _delay_array; }

  std::vector<float>& get_ndelay_array() { return _ndelay_array; }

  std::vector<float>& get_ures_array() { return _ures_array; }

  std::vector<float>& get_ldelay_array() { return _ldelay_array; }

  std::vector<float>& get_beta_array() { return _beta_array; }

  std::vector<float>& get_impulse_array() { return _impulse_array; }

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
  std::vector<std::vector<RctNode*>> _level_to_points;

  std::vector<float> _cap_array;       // levelized cap for gpu speed up.
  std::vector<float> _ncap_array;      // levelized ncap for gpu speed up.
  std::vector<float> _res_array;       // levelized res for gpu speed up.
  std::vector<int> _parent_pos_array;  // levelized parent pos for gpu speed up.
  std::vector<int>
      _children_pos_array;           // levelized children pos for gpu speed up.
  std::vector<float> _load_array;    // levelized load for gpu speed up.
  std::vector<float> _nload_array;   // levelized nload for gpu speed up.
  std::vector<float> _delay_array;   // levelized delay for gpu speed up.
  std::vector<float> _ndelay_array;  // levelized ndelay for gpu speed up.
  std::vector<float> _ures_array;    // levelized ures for gpu speed up.
  std::vector<float> _ldelay_array;  // levelized ldelay for gpu speed up.
  std::vector<float> _beta_array;    // levelized beta for gpu speed up.
  std::vector<float> _impulse_array;  // levelized impulse for gpu speed up.

  float* _gpu_cap_array = nullptr;         // cap located on gpu.
  float* _gpu_ncap_array = nullptr;        // ncap located on gpu.
  float* _gpu_res_array = nullptr;         // res located on gpu.
  int* _gpu_parent_pos_array = nullptr;    // parent pos located on gpu.
  int* _gpu_children_pos_array = nullptr;  // children pos located on gpu.
  float* _gpu_load_array = nullptr;        // load located on gpu.
  float* _gpu_nload_array = nullptr;       // nload located on gpu.
  float* _gpu_delay_array = nullptr;       // delay located on gpu.
  float* _gpu_ndelay_array = nullptr;      // ndelay located on gpu.
  float* _gpu_ures_array = nullptr;        // ures located on gpu.
  float* _gpu_ldelay_array = nullptr;      // ldelay located on gpu.
  float* _gpu_beta_array = nullptr;        // beta located on gpu.
  float* _gpu_impulse_array = nullptr;     // impulse located on gpu.

  void levelizeRcTree(std::queue<RctNode*> bfs_queue);
  void levelizeRcTree();
  void applyDelayDataToArray();

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
  void set_spef_cap_unit(const std::string& spef_cap_unit);
  void set_spef_resistance_unit(const std::string& spef_resistance_unit);
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
  virtual void makeRct(RustSpefNet* spef_net);
  void updateRcTreeInfo();
  virtual void dfsTranverse(RctNode* parent, RctNode& node);
  virtual void checkLoop();
  virtual void breakLoop();
  virtual void updateRcTiming(RustSpefNet* spef_net);

  double load();
  double load(AnalysisMode mode, TransType trans_type);
  double slewImpulse(DesignObject& to, AnalysisMode mode, TransType trans_type);
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
      std::optional<LibCurrentData*> output_current, AnalysisMode mode,
      TransType trans_type);

  virtual std::optional<double> slew(
      DesignObject& to, double from_slew, std::optional<LibCurrentData*> output_current,
      AnalysisMode mode, TransType trans_type);

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

#if CUDA_DELAY
void calc_rc_timing(std::vector<RcNet*> all_nets);
#endif

}  // namespace ista
