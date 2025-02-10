/**
 * @file WaveformApproximation.hh
 * @author LH (liuh0326@163.com)
 * @brief The waveform approximation to calc ceff.
 * @version 0.1
 * @date 2023-05-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <unordered_map>
#include <vector>

#include "Type.hh"

namespace ista {

class RctNode;
class RctEdge;
class RcTree;
class LibArc;
/**
 * @brief Class for pi model.
 *
 */
struct PiModel {
 public:
  double C_near = 0.0;
  double R = 0.0;
  double C_far = 0.0;
};

/**
 * @brief The first three moments of admittance's Laplace expression.
 * Y(s)=y1s+y2s2+y3s3
 */
struct LaplaceMoments {
  double y1 = 0.0;
  double y2 = 0.0;
  double y3 = 0.0;
  LaplaceMoments& operator=(LaplaceMoments* L) {
    y1 = L->y1;
    y2 = L->y2;
    y3 = L->y3;

    return *this;
  }
};

/**
 * @brief waveform approximation, calc ceff cap.
 *
 */
class WaveformApproximation {
 public:
  PiModel reduceRCTreeToPIModel(RctNode* root, double load_nodes_pin_cap_sum);
  LaplaceMoments* calMomentsByDFS(RctNode* the_node);
  LaplaceMoments propagateY(RctEdge* the_edge);

  PiModel calNodePIModel(LaplaceMoments* node_moments);
  // double PiModelToCeff();
  double calInputWaveformThresholdByCeff(
      RcTree& rc_tree, double load_nodes_pin_cap_sum, Eigen::MatrixXd& current,
      Eigen::MatrixXd& time, int input_step_num, TransType trans_type,
      double input_slew, LibArc* lib_arc);
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

  double calVoltageThreshold(Eigen::MatrixXd& T, Eigen::MatrixXd& CU,
                             double Ceff, int num);
  double calCeff1(PiModel& pi_model, double t50, double tr);
  double calCeff2(PiModel& pi_model, double t50);

  auto& getLoadsDelay() { return _load_nodes_delay; }
  auto& getLoadsSlew() { return _load_nodes_slew; }

 private:
  std::map<RctNode*, double> _load_nodes_delay;
  std::map<RctNode*, double> _load_nodes_slew;
};

}  // namespace ista