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
 * @file ArnoldiReduce.hh
 * @author LH (liuh0326@163.com)
 * @brief The arnoldi reduce method use prime algorithm.
 * @version 0.1
 * @date 2023-03-08
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "ElmoreDelayCalc.hh"

using namespace Eigen;
namespace ista {

/**
 * @brief Arnoldi reduce order model.
 *
 */
class ArnoldiROM {
 public:
  struct CircuitParamInit {
    MatrixXd G;
    MatrixXd C;
    VectorXd B;
    VectorXd L;
  };
  struct CircuitParamTrans {
    MatrixXd G_rom;
    MatrixXd C_rom;
    VectorXd B_rom;
    VectorXd L_rom;
  };

  void arnoldiTransfer(const CircuitParamInit& cpi, CircuitParamTrans& cpt,
                       int k);

  // transform conductance matrix.
  MatrixXd GTrans(const MatrixXd& v, const MatrixXd& G) {
    MatrixXd G_rom = v.transpose() * G * v;
    return G_rom;
  }

  // transform capacitance matrix.
  MatrixXd CTrans(const MatrixXd& v, const MatrixXd& C) {
    MatrixXd C_rom = v.transpose() * C * v;
    return C_rom;
  }

  // transform input matrix.
  VectorXd BTrans(const MatrixXd& v, const VectorXd& B) {
    VectorXd B_rom = v.transpose() * B;
    return B_rom;
  }

  // transform ouput matrix.
  VectorXd LTrans(const MatrixXd& v, const VectorXd& L) {
    VectorXd L_rom = v.transpose() * L;
    return L_rom;
  }

  // get Arnoldi orthogonal basis of circuit equation.
  std::optional<MatrixXd> orthogonalBasis(const MatrixXd& G, const MatrixXd& C,
                                          const VectorXd& B, int k);

 private:
  std::optional<MatrixXd> genArnoldiBasis(const MatrixXd& A, const VectorXd& u,
                                          int k);
  MatrixXd blockArnoldi(const MatrixXd& A, const MatrixXd& R, int q, int N);
};

struct PiModel {
 public:
  double C_near;
  double R;
  double C_far;
};

class WaveformApproximation {
 public:
  WaveformApproximation() = default;
  ~WaveformApproximation() = default;

  PiModel reduceRCTreeToPIModel(RcTree& rc_tree, double load_nodes_pin_cap_sum);
  LaplaceMoments* calMomentsByDFS(RctNode* the_node);
  LaplaceMoments propagateY(RctEdge* the_edge);

  PiModel calNodePIModel();
  // double PiModelToCeff();
  double calInputWaveformThresholdByCeff(
      RcTree& rc_tree, double load_nodes_pin_cap_sum, Eigen::MatrixXd& current,
      Eigen::MatrixXd& time, int input_step_num, TransType trans_type,
      double input_slew, LibertyArc* lib_arc);
  double calInputWaveformThresholdByCtotal(double C_total,
                                           Eigen::MatrixXd& current,
                                           Eigen::MatrixXd& time,
                                           int input_step_num);
  void calOutputWaveformThreshold(
      Eigen::MatrixXd& G, Eigen::MatrixXd& C, int iter_num, double tolerence,
      Eigen::MatrixXd& time, int output_step_num, Eigen::MatrixXd& current,
      std::vector<RctNode*>& load_nodes,
      std::unordered_map<RctNode*, unsigned>& nodes_id, TransType trans_type);
  std::map<RctNode*, std::vector<double>> saveLoadsWaveformVoltages(
      int G_size, Eigen::MatrixXd& AX, Eigen::MatrixXd& GX, int iter_num,
      double tolerence, int step_num, Eigen::MatrixXd& cu_interp,
      std::vector<RctNode*>& load_nodes,
      std::unordered_map<RctNode*, unsigned>& nodes_id);
  void calOutputWaveformThresholdAndSlew(
      double step, std::map<RctNode*, std::vector<double>>& load_voltages,
      double slew_coefficient, TransType trans_type);

  double calVoltageThreshold(MatrixXd& T, MatrixXd& CU, double Ceff, int num);
  double calCeff1(PiModel& pi_model, double t50, double tr);
  double calCeff2(PiModel& pi_model, double t50);

  auto& getLoadsDelay() { return _load_nodes_delay; }
  auto& getLoadsSlew() { return _load_nodes_slew; }

 private:
  std::map<RctNode*, double> _load_nodes_delay;
  std::map<RctNode*, double> _load_nodes_slew;
};

}  // namespace ista