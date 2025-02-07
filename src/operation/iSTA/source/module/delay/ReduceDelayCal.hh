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
 * @file ReduceDelayCal.hh
 * @author LH (liuh0326@163.com)
 * @brief Calc the delay by ccs model.
 * @version 0.1
 * @date 2021-07-04
 */

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <mutex>
#include <optional>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ElmoreDelayCalc.hh"
#include "Type.hh"
#include "WaveformInfo.hh"

using namespace Eigen;

namespace ista {

/**
 * @brief Reduce the rc tree, only reserve the driver to load path.
 *
 */
class ReducedPath {
 public:
  void insertPathNode(RctNode* the_node) { _path_nodes.push_back(the_node); }
  void insertPathEdge(RctEdge* the_edge) { _path_edges.push_back(the_edge); }

  auto& get_path_nodes() { return _path_nodes; }
  auto& get_path_edges() { return _path_edges; }

  void popNode() { _path_nodes.pop_back(); }
  void popEdge() { _path_edges.pop_back(); }

  void recordPathEdge() {
    for (auto* path_edge : _path_edges) {
      _all_reduced_edges.insert(path_edge);
    }
  }

  void deleteNoCapNode(RcTree& rc_tree, const std::set<RctNode*>& pin_nodes);
  void makeTree(RcTree& rc_tree, const std::set<RctNode*>& pin_nodes);

 private:
  std::vector<RctNode*> _path_nodes;
  std::vector<RctEdge*> _path_edges;

  std::unordered_set<RctEdge*>
      _all_reduced_edges;  //!< record the all reserverd reduced edge.
};

/**
 * @brief Net for arnoldi calc.
 *
 */
class ArnoldiNet : public RcNet {
 public:
  ArnoldiNet();
  explicit ArnoldiNet(Net* net);
  ~ArnoldiNet() override = default;

  bool getPathByDFS(RctNode* the_node, RctNode* root_node,
                    ReducedPath& reduced_path);
  void makeRcTreeReduce();

  void insertNodeID(RctNode* node, unsigned id) {
    _node_to_id[node] = id;
    _id_to_node[id] = node;
  }
  unsigned getNodeID(RctNode* node) { return _node_to_id[node]; }
  auto* getIDNode(unsigned id) { return _id_to_node[id]; }
  void assignRcNodeID();

  void updateRcTiming(RustSpefNet* spef_net) override;

  void set_nodal_caps(std::vector<double>&& nodal_caps) {
    _nodal_caps = std::move(nodal_caps);
  }

  LibArc* get_lib_arc() { return _lib_arc; }
  void set_lib_arc(LibArc* lib_arc) { _lib_arc = lib_arc; }

  std::optional<double> calcDelay(const VectorXd& driver_waveform,
                                  const VectorXd& load_waveform,
                                  double step_time_ns, DesignObject* pin);

  std::optional<double> calcSlew(const VectorXd& waveform, double step_time_ns,
                                 DesignObject* pin);

  std::optional<double> delay(Waveform& driver_waveform,
                              Waveform& node_waveform, DesignObject* pin);

  std::optional<double> slew(Waveform& node_waveform, DesignObject* pin);

  std::optional<std::pair<double, MatrixXd>> getDelay(
      std::function<std::vector<double>(double, double, int)>&& get_current,
      double start_time, double end_time, int num_sim_point,
      AnalysisMode analysis_mode, TransType trans_type, DesignObject* pin);

  std::optional<double> getSlew(
      std::function<std::vector<double>(double, double, int)>&& get_current,
      double start_time, double end_time, int num_sim_point,
      AnalysisMode analysis_mode, TransType trans_type, DesignObject* pin);

  std::optional<std::pair<double, MatrixXd>> delay(
      DesignObject& to, double from_slew,
      std::optional<LibCurrentData*> output_current, AnalysisMode mode,
      TransType trans_type) override;
  std::optional<double> slew(DesignObject& to, double from_slew,
                             std::optional<LibCurrentData*> output_current,
                             AnalysisMode mode, TransType trans_type) override;

  void set_is_debug(bool is_debug) { _is_debug = is_debug; }
  bool isDebug() const { return _is_debug; }

  void set_is_reduce(bool is_reduce) { _is_reduce = is_reduce; }
  bool isReduce() const { return _is_reduce; }

 private:
  MatrixXd calcDelayAndSlew(
      std::function<std::vector<double>(double, double, int)>&& get_current,
      double start_time, double end_time, int num_sim_point,
      AnalysisMode analysis_mode, TransType trans_type);

  void constructResistanceAndCapMatrix(AnalysisMode analysis_mode,
                                       TransType trans_type);
  unsigned constructArnoldiOrthogonalBasis();
  void reduceRCEquation();

  auto constructRCEquation(const MatrixXd& cap_matrix,
                           const MatrixXd& conductances_matrix,
                           const VectorXd& input_vec);
  std::vector<VectorXd> solveRCEquation(
      std::function<std::vector<double>(double, double, int)>&& get_current,
      double start_time, double end_time, int num_sim_point, MatrixXd& diag,
      MatrixXd& B);

  unsigned getPinNodeId(DesignObject* pin) {
    auto it = std::find_if(
        _node_to_id.begin(), _node_to_id.end(),
        [pin](const auto& pair) { return (pair.first->get_obj() == pin); });
    return it->second;
  }

  VectorXd getOutputVector(unsigned id);

  std::optional<double> getWaveformPointTime(const VectorXd& waveform,
                                             double threshold_point,
                                             double step_time);

  std::unordered_map<RctNode*, unsigned> _node_to_id;
  std::unordered_map<unsigned, RctNode*> _id_to_node;

  LibArc* _lib_arc{nullptr};

  std::vector<double> _nodal_caps;  //!< The nodal cap matrix.
  MatrixXd _conductances_matrix;    //!< The conductance matrix.
  MatrixXd _cap_matrix;             //!< The cap matrix.
  VectorXd _input_vec;              //!< The current input vec.

  MatrixXd _arnoldi_basis;               //!< The arnoldi basis for reduce.
  MatrixXd _reduce_conductances_matrix;  //!< The reduce conductance matrix.
  MatrixXd _reduce_cap_matrix;           //!< The reduce cap matrix.
  MatrixXd _reduce_input_vec;            //!< The reduce input select vector.

  std::optional<std::tuple<MatrixXd, MatrixXd, MatrixXd>>
      _diag_B_W;  // The RC equation matrix.

  std::mutex _calc_mutex;

  unsigned _is_debug : 1 = 0;
  unsigned _is_reduce : 1 = 0;  // default reduce.
  unsigned _reserved : 30 = 0;
};

}  // namespace ista