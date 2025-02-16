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
 * @file ReduceDelayCal.cc
 * @author LH (liuh0326@163.com)
 * @brief Calc the delay by ccs model.
 * @version 0.1
 * @date 2021-07-04
 */

// #include <gperftools/heap-checker.h>
// #include <gperftools/profiler.h>

#include "ReduceDelayCal.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_set>

#include "Reduce.hh"
#include "Type.hh"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Instance.hh"
#include "solver/Interpolation.hh"
#include "utility/EigenMatrixUtility.hh"
#include "netlist/Net.hh"
#include "netlist/Pin.hh"
#include "netlist/Port.hh"

using namespace Eigen;

namespace ista {

/**
 * @brief Delete zero cap node.
 *
 * @param rc_tree
 */
void ReducedPath::deleteNoCapNode(RcTree& rc_tree,
                                  const std::set<RctNode*>& pin_nodes) {
  Vector<RctNode*> to_be_removed_nodes;
  for (auto& [node_name, rc_node] : rc_tree.get_nodes()) {
    if (rc_node.cap() == 0.0) {
      // only delete the internal direct node.
      if (((rc_node.get_fanin().size() == 2) &&
           (rc_node.get_fanout().size() == 2)) &&
          std::find_if(pin_nodes.begin(), pin_nodes.end(),
                       [&rc_node](RctNode* pin_node) {
                         return pin_node->get_name() == rc_node.get_name();
                       }) == pin_nodes.end()) {
        to_be_removed_nodes.push_back(&rc_node);
      } else {
        rc_node.setCap(1e-6);  // set the zero cap to be 1e-6
      }
    }
  }

  auto delete_inner_node = [&rc_tree](auto* rc_node) {
    /*get the front and back node*/
    std::optional<std::string> node1_name;
    std::optional<std::string> node2_name;

    double res = 0.0;
    for (auto* fanin_edge : rc_node->get_fanin()) {
      if (!node1_name) {
        node1_name = fanin_edge->get_from().get_name();
      } else {
        node2_name = fanin_edge->get_from().get_name();
      }

      res += fanin_edge->get_res();
    }

    rc_tree.insertSegment(*node1_name, *node2_name, res);
    rc_tree.removeNode(rc_node);
  };

  for (auto* to_be_removed_node : to_be_removed_nodes) {
    delete_inner_node(to_be_removed_node);
  }
}

/**
 * @brief Construct the rc tree accord the reduced path.
 *
 * @param rc_tree
 */
void ReducedPath::makeTree(RcTree& rc_tree,
                           const std::set<RctNode*>& pin_nodes) {
  for (auto* reduce_edge : _all_reduced_edges) {
    auto& node1 = reduce_edge->get_from();
    rc_tree.insertNode(node1.get_name(), node1.get_cap());
    auto& node2 = reduce_edge->get_to();
    rc_tree.insertNode(node2.get_name(), node2.get_cap());
    double res = reduce_edge->get_res();
    rc_tree.insertSegment(node1.get_name(), node2.get_name(), res);
  }

  deleteNoCapNode(rc_tree, pin_nodes);
}

ArnoldiNet::ArnoldiNet(Net* net) : RcNet(net) {}

/**
 * @brief Tranverse all driver to load path edge.
 *
 * @param the_node
 * @param load_nodes
 * @param reduced_path
 */
bool ArnoldiNet::getPathByDFS(RctNode* the_node, RctNode* load_node,
                              ReducedPath& reduced_path) {
  the_node->set_is_visited(true);
  if (the_node == load_node) {
    reduced_path.recordPathEdge();
    return true;
  }

  bool is_ok = false;
  for (auto* fanout_edge : the_node->get_fanout()) {
    auto& fanout_node = fanout_edge->get_to();
    if (fanout_node.isVisited()) {
      continue;
    }
    reduced_path.insertPathEdge(fanout_edge);
    if (getPathByDFS(&fanout_node, load_node, reduced_path)) {
      is_ok = true;
      break;
    }
    reduced_path.popEdge();
  }
  return is_ok;
}

/**
 * @brief Reduce the unused node.
 *
 */
void ArnoldiNet::makeRcTreeReduce() {
  if (_rct.index() == 0) {
    return;
  }

  /*check and break loop*/
  checkLoop();

  /*tranverse and reduce path*/
  auto load_nodes = getLoadNodes();
  auto& rc_tree = std::get<RcTree>(_rct);
  // rc_tree.printGraphViz(); //for debug

  ReducedPath reduced_path;
  bool is_ok = false;
  for (auto* load_node : load_nodes) {
    is_ok = getPathByDFS(rc_tree.get_root(), load_node, reduced_path);
    LOG_FATAL_IF(!is_ok) << "no path from root to load node";
    rc_tree.resetNodeVisit();
  }

  auto pin_nodes = std::move(load_nodes);
  pin_nodes.insert(rc_tree.get_root());

  RcTree reduce_rc_tree;
  reduced_path.makeTree(reduce_rc_tree, pin_nodes);
  swap(rc_tree, reduce_rc_tree);

  if (rc_tree.get_node_num() == 0) {
    EmptyRct empty_tree{.load = 0.0};
    _rct = std::move(empty_tree);
  }

  /*identify root and obj*/
  updateRcTreeInfo();
}

/**
 * @brief Assign rc node ID for matrix index.
 *
 */
void ArnoldiNet::assignRcNodeID() {
  auto& rct = std::get<RcTree>(_rct);

  unsigned id = 0;
  auto* root_node = rct.get_root();
  insertNodeID(root_node, id);

  FOREACH_RCTREE_NODE(rct, name, node) {
    if (&node != root_node) {
      insertNodeID(&node, ++id);
    }
  }
}

/**
 * @brief Update RC Timing, calc delay and slew.
 *
 * @param parser
 */
void ArnoldiNet::updateRcTiming(RustSpefNet* spef_net) {
  makeRct(spef_net);
  updateRcTreeInfo();
  makeRcTreeReduce();

  if (_rct.index() != 0) {
    auto& rct = std::get<RcTree>(_rct);
    rct.initData();
    if (rct._root) {
      rct.updateLoad(nullptr, rct.get_root());
      rct.updateDelay(nullptr, rct.get_root());
    }

    assignRcNodeID();
  }
}

/**
 * @brief Store the resistances of each segment and the capacitance of each
 * nodal in vector container.
 *
 * @param analysis_mode
 * @param trans_type
 */
void ArnoldiNet::constructResistanceAndCapMatrix(AnalysisMode analysis_mode,
                                                 TransType trans_type) {
  constexpr unsigned use_reduce_node_num = 10;

  auto& rct = std::get<RcTree>(_rct);
  auto node_num = rct.get_node_num();

  // reduce when node num beyond reduce node num.
  if (node_num > use_reduce_node_num) {
    // not use reduce now.
    // set_is_reduce(true);
  }

  // construct the matrix C and diagonal G.
  std::vector<double> nodal_caps(node_num);
  MatrixXd conductances(node_num, node_num);
  conductances.setZero();

  FOREACH_RCTREE_NODE(rct, name, rc_node) {
    // construct the matrix C.
    double cap = rc_node.get_cap(analysis_mode, trans_type);
    unsigned node_id = getNodeID(&rc_node);
    nodal_caps[node_id] = PF_TO_F(cap);  // convert PF to standard F.

    // construct the G diagonal part.
    double conductance = 0.0;
    FOREACH_RCNODE_FANIN_EDGE(&rc_node, fanin_edge) {
      conductance += fanin_edge->getG(analysis_mode, trans_type);
    }
    conductances(node_id, node_id) =
        conductance;  // conductance is resistance inv.
  }

  set_nodal_caps(std::move(nodal_caps));

  // construct the matrix G other part.
  for (decltype(node_num) i = 0; i < node_num; ++i) {
    auto* rc_node = getIDNode(i);
    FOREACH_RCNODE_FANOUT_EDGE(rc_node, fanout_edge) {
      auto& fanout_rc_node = fanout_edge->get_to();
      unsigned node_id = getNodeID(&fanout_rc_node);
      conductances(i, node_id) = -(fanout_edge->getG(
          analysis_mode, trans_type));  // get the opposite val.
    }
  }

  _conductances_matrix = conductances;

  // consturct nodal cap matrix.
  auto cap_size = _nodal_caps.size();
  VectorXd nodal_vec(cap_size);
  for (decltype(cap_size) i = 0; i < cap_size; ++i) {
    nodal_vec(i) = _nodal_caps[i];
  }

  _cap_matrix = nodal_vec.asDiagonal();

  // construct input vec.
  VectorXd input_vec(_nodal_caps.size());
  input_vec.setZero();
  input_vec(0) = 1.0;

  _input_vec = input_vec;

  DVERBOSE_VLOG(1) << "conductances\n" << _conductances_matrix;
  DVERBOSE_VLOG(1) << "cap_matrix\n" << _cap_matrix;
  DVERBOSE_VLOG(1) << "input_vec\n" << _input_vec;
}

/**
 * @brief construct arnoldi orthogonal basis use
 *
 */
unsigned ArnoldiNet::constructArnoldiOrthogonalBasis() {
  constexpr int arnoldi_reduce_order = 3;
  ArnoldiROM arnoldi_rom;
  auto arnoldi_basis = arnoldi_rom.orthogonalBasis(
      _conductances_matrix, _cap_matrix, _input_vec, arnoldi_reduce_order);

  if (!arnoldi_basis) {
    LOG_ERROR << "no suitable arnoldi basis.";
    return 0;
  }

  _arnoldi_basis = arnoldi_basis.value();
  DVERBOSE_VLOG(1) << "arnoldi_basis\n" << _arnoldi_basis;
  return 1;
}

/**
 * @brief Reduce rc equation use arnoldi basis.
 *
 */
void ArnoldiNet::reduceRCEquation() {
  ArnoldiROM arnoldi_rom;

  _reduce_conductances_matrix =
      arnoldi_rom.GTrans(_arnoldi_basis, _conductances_matrix);
  _reduce_cap_matrix = arnoldi_rom.CTrans(_arnoldi_basis, _cap_matrix);
  _reduce_input_vec = arnoldi_rom.BTrans(_arnoldi_basis, _input_vec);

  DVERBOSE_VLOG(1) << "reduce_conductances_matrix\n"
                   << _reduce_conductances_matrix;
  DVERBOSE_VLOG(1) << "reduce_cap_matrix\n" << _reduce_cap_matrix;
  DVERBOSE_VLOG(1) << "reduce_input_vec\n" << _reduce_input_vec;
}

/**
 * @brief Construct the RC equation.
 *
 *  please reference paper <<An Efficient Method for Fast Delay and SI
 * Calculation Using Current Source Models>> but dont agree with the paper, we
 * use inv cap , not use inv G.
 */
auto ArnoldiNet::constructRCEquation(const MatrixXd& cap_matrix,
                                     const MatrixXd& conductances_matrix,
                                     const VectorXd& input_vec) {
  auto cap_inv = cap_matrix.inverse();

  // LOG_INFO << "cap matrix:\n" << cap_matrix;
  // LOG_INFO << "cap inv:\n" << cap_inv;
  // LOG_INFO << "conductances:\n" << (*_conductances);

  auto cap_inv_dot_conductances = cap_inv * conductances_matrix;
  // LOG_INFO << cap_inv_dot_conductances;

  EigenSolver<MatrixXd> es(cap_inv_dot_conductances);
  MatrixXd diag = es.pseudoEigenvalueMatrix();

  const auto& W = es.pseudoEigenvectors();
  auto W_inv = W.inverse();

  MatrixXd B = W_inv * cap_inv * input_vec;

  // LOG_INFO << "diag:\n" << diag;
  // LOG_INFO << "W:\n" << W;
  // LOG_INFO << "B:\n" << B;

  return std::make_tuple(diag, B, W);
}

/**
 * @brief Solve rc circuit equation use newton method.The rc circuit is :
 * different(V) + diag*V= B.
 * V is not origin V, is W_inv * V.
 * @param get_current the function to get input current.
 * @param start_time simulate start time.
 * @param end_time simulate end time.
 * @param num_sim_point num of sim point.
 * @param diag the diag matrix of C inv dot G.
 * @param B the input vector, (W_inv * cap_inv * input_vec).
 * @return std::vector<VectorXd>
 */
std::vector<VectorXd> ArnoldiNet::solveRCEquation(
    std::function<std::vector<double>(double, double, int)>&& get_current,
    double start_time, double end_time, int num_sim_point, MatrixXd& diag,
    MatrixXd& B) {
  const double precision = 1e-6;
  const unsigned max_iter = 1;
  const double mA_to_A = 1e-3;
  const double ns_to_s = 1e-9;

  double step_time_ns = (end_time - start_time) / (num_sim_point - 1);
  double step_time = step_time_ns * ns_to_s;

  auto F = [step_time, &diag, &B](VectorXd& V_i, VectorXd& V_i_1,
                                  double current) -> MatrixXd {
    MatrixXd func_val =
        diag * V_i_1 + (1 / step_time) * (V_i_1 - V_i) - B * current;
    return func_val;
  };

  auto F_derivative = [step_time, &diag]() -> MatrixXd {
    MatrixXd unit_vec(diag.diagonalSize(), diag.diagonalSize());
    unit_vec.setOnes();

    MatrixXd derivate = diag + (1 / step_time) * unit_vec;

    return derivate;
  };

  auto Newton_method = [&F_derivative, &F](VectorXd& V_i, VectorXd& V_i_1,
                                           double precision, unsigned max_iter,
                                           double current) {
    for (unsigned iter = 0; iter < max_iter; ++iter) {
      MatrixXd derivate = F_derivative();
      // LOG_INFO << "\n" << derivate;
      MatrixXd derivate_inv = F_derivative().inverse();
      MatrixXd func_val = F(V_i, V_i_1, current);

      // LOG_INFO << "\n" << derivate_inv;
      // LOG_INFO << "\n" << func_val;

      MatrixXd new_V_i_1 = V_i_1 - derivate_inv * func_val;

      VectorXd tmp = new_V_i_1.col(0);
      V_i_1 = tmp;
    }
  };

  auto currents = get_current(start_time, end_time, num_sim_point);

  // for debug
  if (0 && Str::equal(_net->get_name(), "clk_core_1936")) {
    // DVERBOSE_VLOG(1) << "conductances\n" << _conductances_matrix;
    // DVERBOSE_VLOG(1) << "cap_matrix\n" << _cap_matrix;
    // DVERBOSE_VLOG(1) << "input_vec\n" << _input_vec;

    // DVERBOSE_VLOG(1) << "currents\n";
    // for (double current_mA : currents) {
    //   DVERBOSE_VLOG(1) << " " << current_mA;
    // }

    std::ofstream file("matrix.txt");
    file << "conductances\n" << _conductances_matrix << "\n";
    file << "cap_matrix\n" << _cap_matrix << "\n";
    file << "input_vec\n" << _input_vec << "\n";
    file << "currents\n";
    for (double current_mA : currents) {
      file << " " << current_mA / 1000.0;
    }
    file.close();
  }

  VectorXd V_i(diag.diagonalSize());
  V_i.setZero();

  VectorXd V_i_1 = V_i;

  std::vector<VectorXd> V_waveform;
  for (double current_mA : currents) {
    double current = current_mA * mA_to_A;
    Newton_method(V_i, V_i_1, precision, max_iter, current);
    V_waveform.emplace_back(V_i);
    V_i = V_i_1;
  }

  return V_waveform;
}

/**
 * @brief get output vector.
 *
 * @param id
 * @return VectorXd
 */
VectorXd ArnoldiNet::getOutputVector(unsigned id) {
  VectorXd output_vec(_input_vec.rows());
  output_vec.setZero();
  output_vec(id) = 1.0;

  // for reduce output vector.
  if (isReduce()) {
    ArnoldiROM arnoldi_rom;
    output_vec = arnoldi_rom.LTrans(_arnoldi_basis, output_vec);
  }
  return output_vec;
}

/**
 * @brief Use trapezoidal method to calculate the net delay and slew. if the
delay and slew has been calculated,do not need to calculate again.
 *
 * @param current
 * @param sim_total_time
 * @param num_sim_point
 * @param trans_type
 */
MatrixXd ArnoldiNet::calcDelayAndSlew(
    std::function<std::vector<double>(double, double, int)>&& get_current,
    double start_time, double end_time, int num_sim_point,
    AnalysisMode analysis_mode, TransType trans_type) {
  // HeapLeakChecker heap_checker("test_foo");
  {
    if (!_diag_B_W) {
      std::lock_guard<std::mutex> lk(_calc_mutex);

      if (!_diag_B_W) {
        constructResistanceAndCapMatrix(analysis_mode, trans_type);
        if (!isReduce()) {
          _diag_B_W = constructRCEquation(_cap_matrix, _conductances_matrix,
                                          _input_vec);
        } else {
          if (constructArnoldiOrthogonalBasis()) {
            reduceRCEquation();
            _diag_B_W = constructRCEquation(_reduce_cap_matrix,
                                            _reduce_conductances_matrix,
                                            _reduce_input_vec);
          } else {
            set_is_reduce(false);
            _diag_B_W = constructRCEquation(_cap_matrix, _conductances_matrix,
                                            _input_vec);
          }
        }
      }
    }

    auto [diag, B, W] = *_diag_B_W;

    DVERBOSE_VLOG(1) << "diag\n" << diag;
    DVERBOSE_VLOG(1) << "W\n" << W;
    DVERBOSE_VLOG(1) << "B\n" << B;

    // Get back Euler integration
    auto V_waveform = solveRCEquation(std::move(get_current), start_time,
                                      end_time, num_sim_point, diag, B);
    MatrixXd V_matrix(V_waveform[0].size(), V_waveform.size());
    int i = 0;
    for (auto& V : V_waveform) {
      // LOG_INFO << "V raw\n" << V;
      // LOG_INFO << "V\n" << W * V;
      V_matrix.col(i++) =
          W * V;  // get the origin V, V is W_inv * origin V.
                  // every column is one time voltage of every point.
    }
    DVERBOSE_VLOG(1) << "V Matrix \n" << V_matrix;

    return V_matrix;
  }

  // if (!heap_checker.NoLeaks()) assert(NULL == "heap memory leak");
}

/**
 * @brief Accord the wavefrom, get the point time of threshold.
 *
 * @param waveform
 * @param threshold_point
 * @param step_time
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::getWaveformPointTime(const VectorXd& waveform,
                                                       double threshold_point,
                                                       double step_time) {
  double time1;
  double time2;
  double v1;
  double v2;
  bool is_found = false;
  auto num_point = waveform.rows();
  for (int i = 0; i < num_point; ++i) {
    if (waveform(i) >= threshold_point) {
      time1 = i * step_time;
      v1 = waveform(i);
      time2 = (i - 1) * step_time;
      v2 = waveform(i - 1);
      is_found = true;
      break;
    }
  }

  std::optional<double> time;
  if (is_found) {
    time = LinearInterpolate(v2, v1, time2, time1, threshold_point);
  }
  return time;
}

/**
 * @brief Calc delay accord waveform.
 *
 * @param V_matrix
 * @param node_id
 * @param step_time_ns
 * @param analysis_mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::calcDelay(const VectorXd& driver_waveform,
                                            const VectorXd& load_waveform,
                                            double step_time_ns,
                                            DesignObject* pin) {
  auto* the_lib = pin->get_own_instance()->get_inst_cell()->get_owner_lib();
  double nom_voltage = the_lib->get_nom_voltage();

  const double input_threshold_pct_rise =
      the_lib->get_input_threshold_pct_rise() * nom_voltage;
  const double output_threshold_pct_rise =
      the_lib->get_output_threshold_pct_rise() * nom_voltage;

  std::optional<double> delay;

  auto driver_time = getWaveformPointTime(
      driver_waveform, input_threshold_pct_rise, step_time_ns);

  auto load_time = getWaveformPointTime(
      load_waveform, output_threshold_pct_rise, step_time_ns);

  if (load_time && driver_time) {
    auto delay_ns = *load_time - *driver_time;
    delay = NS_TO_PS(delay_ns);

    VERBOSE_VLOG_IF_EVERY_N(0, isReduce(), 2000)
        << "calculate net: " << get_net()->getFullName() << " "
        << " delay is " << *delay << " ps";

    if (isDebug() && IsDoubleEqual(delay_ns, 0.0)) {
      LOG_INFO_EVERY_N(100) << "the net " << _net->get_name()
                            << " driver time: " << *driver_time << " "
                            << " load time: " << *load_time << " ";
    }

  } else {
    if (isDebug()) {
      if (!driver_time) {
        DVERBOSE_VLOG(1) << "driver waveform \n" << driver_waveform;
      }

      if (!load_time) {
        DVERBOSE_VLOG(1) << "load waveform \n" << load_waveform;
      }
    }
  }

  return delay;
}

/**
 * @brief Get rct node delay.
 *
 * @param node
 * @param analysis_mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::delay(Waveform& driver_waveform,
                                        Waveform& node_waveform,
                                        DesignObject* pin) {
  return calcDelay(driver_waveform.get_waveform_vector(),
                   node_waveform.get_waveform_vector(),
                   driver_waveform.get_step_time_ns(), pin);
}

/**
 * @brief inner calc delay use ccs model.
 *
 * @param get_current
 * @param start_time
 * @param end_time
 * @param num_sim_point
 * @param analysis_mode
 * @param trans_type
 * @param pin
 * @return std::optional<double>
 */
std::optional<std::pair<double, MatrixXd>> ArnoldiNet::getDelay(
    std::function<std::vector<double>(double, double, int)>&& get_current,
    double start_time, double end_time, int num_sim_point,
    AnalysisMode analysis_mode, TransType trans_type, DesignObject* pin) {
  // std::lock_guard<std::mutex> lk(_calc_mutex);
  MatrixXd V_matrix =
      calcDelayAndSlew(std::move(get_current), start_time, end_time,
                       num_sim_point, analysis_mode, trans_type);

  double step_time_ns = (end_time - start_time) / (num_sim_point - 1);

  unsigned load_id = getPinNodeId(pin);
  VectorXd driver_output_vec = getOutputVector(0);
  VectorXd load_output_vec = getOutputVector(load_id);

  // every row is voltage change of the point.
  VectorXd driver_waveform = driver_output_vec.transpose() * V_matrix;
  VectorXd load_waveform = load_output_vec.transpose() * V_matrix;

  DVERBOSE_VLOG(1) << "driver_waveform\n" << driver_waveform;
  DVERBOSE_VLOG(1) << "load_waveform\n" << load_waveform;

  auto delay = calcDelay(driver_waveform, load_waveform, step_time_ns, pin);

  if (delay) {
    return std::make_pair(delay.value(), V_matrix);
  }

  return std::nullopt;
}

/**
 * @brief Calc slew accord waveform.
 *
 * @param waveform
 * @param step_time_ns
 * @param analysis_mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::calcSlew(const VectorXd& waveform,
                                           double step_time_ns,
                                           DesignObject* pin) {
  auto* the_lib = pin->get_own_instance()->get_inst_cell()->get_owner_lib();
  double nom_voltage = the_lib->get_nom_voltage();

  const double slew_lower_threshold_pct_rise_voltage =
      the_lib->get_slew_lower_threshold_pct_rise() * nom_voltage;
  const double slew_upper_threshold_pct_rise_voltage =
      the_lib->get_slew_upper_threshold_pct_rise() * nom_voltage;

  auto lower_threshold_time = getWaveformPointTime(
      waveform, slew_lower_threshold_pct_rise_voltage, step_time_ns);

  auto upper_threshold_time = getWaveformPointTime(
      waveform, slew_upper_threshold_pct_rise_voltage, step_time_ns);

  std::optional<double> slew;
  if (upper_threshold_time && lower_threshold_time) {
    double slew_ns = *upper_threshold_time - *lower_threshold_time;
    slew = NS_TO_PS(slew_ns);

    VERBOSE_VLOG_IF_EVERY_N(0, isReduce(), 2000)
        << "calculate net: " << get_net()->getFullName() << " "
        << " slew is " << *slew << " ps";
  }

  return slew;
}

/**
 * @brief Get rct node of slew.
 *
 * @param node
 * @param analysis_mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::slew(Waveform& node_waveform,
                                       DesignObject* pin) {
  auto slew = calcSlew(node_waveform.get_waveform_vector(),
                       node_waveform.get_step_time_ns(), pin);
  return slew;
}

/**
 * @brief Return slew. if the delay and slew has been calculated,do not
 * need to calculate again.
 *
 * @param current
 * @param sim_total_time
 * @param num_sim_point
 * @param trans_type
 * @param pin
 * @return double
 */
std::optional<double> ArnoldiNet::getSlew(
    std::function<std::vector<double>(double, double, int)>&& get_current,
    double start_time, double end_time, int num_sim_point,
    AnalysisMode analysis_mode, TransType trans_type, DesignObject* pin) {
  std::optional<double> slew;

  {
    // std::lock_guard<std::mutex> lk(_calc_mutex);
    DVERBOSE_VLOG(1) << "calculate net: " << get_net()->getFullName();

    MatrixXd V_matrix =
        calcDelayAndSlew(std::move(get_current), start_time, end_time,
                         num_sim_point, analysis_mode, trans_type);

    double step_time_ns = (end_time - start_time) / (num_sim_point - 1);

    unsigned load_id = getPinNodeId(pin);
    VectorXd load_output_vec = getOutputVector(load_id);
    DVERBOSE_VLOG(1) << "output vec\n" << load_output_vec;
    VectorXd load_waveform = load_output_vec.transpose() * V_matrix;

    DVERBOSE_VLOG(1) << "load_waveform\n" << load_waveform;

    slew = calcSlew(load_waveform, step_time_ns, pin);
  }

  return slew;
}

/**
 * @brief Get the net delay used the Arnoldi calc method.
 *
 * @param to The load pin.
 * @param from_slew The net driver output slew.
 * @param output_current The driver output current if use ccs else null.
 * @param mode The max/min mode.
 * @param trans_type The rise/fall transition type.
 * @return std::optional<double>
 */
std::optional<std::pair<double, MatrixXd>> ArnoldiNet::delay(
    DesignObject& to, double from_slew,
    std::optional<LibCurrentData*> output_current, AnalysisMode mode,
    TransType trans_type) {
  if (output_current) {
    auto [total_time, num_points] =
        (*output_current)->getSimulationTotalTimeAndNumPoints();

    auto get_current = [&output_current](
                           double start_time, double end_time,
                           int num_sim_point) -> std::vector<double> {
      std::optional<LibCurrentSimuInfo> simu_info =
          LibCurrentSimuInfo{start_time, end_time, num_sim_point};

      return (*output_current)->getOutputCurrent(simu_info);
    };

    return getDelay(std::move(get_current), 0, total_time, num_points, mode,
                    trans_type, &to);
  }

  return std::nullopt;
}

/**
 * @brief Get the net slew used the Arnoldi calc method.
 *
 * @param to The load pin.
 * @param from_slew The net driver output slew.
 * @param output_current The driver output current if use ccs else null.
 * @param mode The max/min mode.
 * @param trans_type The rise/fall transition type.
 * @return std::optional<double>
 */
std::optional<double> ArnoldiNet::slew(
    DesignObject& to, double from_slew, std::optional<LibCurrentData*> output_current,
    AnalysisMode mode, TransType trans_type) {
  // static int prof_count = 0;
  // if (prof_count == 0) {
  //   ProfilerStart("STAPerf.prof");
  // }

  if (output_current) {
    auto [total_time, num_points] =
        (*output_current)->getSimulationTotalTimeAndNumPoints();

    auto get_current = [&output_current](
                           double start_time, double end_time,
                           int num_sim_point) -> std::vector<double> {
      std::optional<LibCurrentSimuInfo> simu_info =
          LibCurrentSimuInfo{start_time, end_time, num_sim_point};

      return (*output_current)->getOutputCurrent(simu_info);
    };

    return getSlew(std::move(get_current), 0, total_time, num_points, mode,
                   trans_type, &to);
  }
  // prof_count++;
  // if (prof_count > 400000) {
  //   ProfilerStop();
  //   exit(1);
  // }

  return std::nullopt;
}

}  // namespace ista
