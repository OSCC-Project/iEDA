/**
 * @file ArnoldiReduce.cc
 * @author LH (you@domain.com)
 * @brief The arnoldi reduce method implemention use prime algorithm.
 * @version 0.1
 * @date 2023-03-08
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "ArnoldiReduce.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "log/Log.hh"
#include "utility/EigenMatrixUtility.hh"

using namespace Eigen;
namespace ista {

constexpr static double E = 2.7183;
/**
 * @brief use Arnoldi process based Krylov subspace to calculate a set of
 * orthonormal vectors.
 * @param A n*n matrix
 * @param u n*1 initial vector
 * @param k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 */
std::optional<MatrixXd> ArnoldiROM::genArnoldiBasis(const MatrixXd& A,
                                                    const VectorXd& u, int k) {
  constexpr double eplison = 1e-19;
  auto size = A.rows();
  MatrixXd v(size, k + 1);
  MatrixXd h(k + 1, k);
  VectorXd w(size);
  v.col(0) = u / (u.norm());
  for (int j = 0; j < k; ++j) {
    w = A * v.col(j);
    for (int i = 0; i <= j; ++i) {
      h(i, j) = v.col(i).transpose() * w;
      w = w - h(i, j) * v.col(i);
    }
    h(j + 1, j) = w.norm();
    if (IsDoubleEqual(h(j + 1, j), 0.0, eplison)) {
      LOG_ERROR << "Denominator " << h(j + 1, j) << " is zero";
      return std::nullopt;
    }
    v.col(j + 1) = w / h(j + 1, j);
  }
  return v;
}
/**
 * @brief use Block Arnoldi Algorithm based block Krylov subspace to calculate a
 * set of orthonormal vectors.
 *
 * @param A n*n matrix
 * @param R
 * @param q the column number of orthonormal vectors
 * @param N the number of terminals(ports)
 * @return MatrixXd
 */
MatrixXd ArnoldiROM::blockArnoldi(const MatrixXd& A, const MatrixXd& R, int q,
                                  int N) {
  HouseholderQR<MatrixXd> qr0;
  HouseholderQR<MatrixXd> qrj;
  qr0.compute(R);
  MatrixXd v0 = qr0.householderQ();  // Q
  MatrixXd vj;
  std::vector<MatrixXd*> V_basis;
  std::vector<std::vector<MatrixXd*>> H;
  *V_basis[0] = v0;
  MatrixXd w;
  MatrixXd v_trans;
  MatrixXd V(q, q);
  int k = q / N;
  for (int j = 1; j <= k; ++j) {
    w = A * (*V_basis[j - 1]);
    for (int i = 1; i <= j; ++j) {
      v_trans = V_basis[j - i]->transpose();
      *(H[j - i][j - 1]) = v_trans * w;
      w = w - *V_basis[j - i] * (*(H[j - i][j - 1]));
    }
    qrj.compute(w);
    vj = qrj.householderQ();
    *V_basis[j] = vj;
  }
  int basis_block_num = V_basis.size();
  int block_size = 0;
  int col = 0;
  for (int m = 0; m < basis_block_num; ++m) {
    block_size = V_basis[m]->cols();
    for (int n = 0; n < block_size; ++n) {
      V.col(col) = V_basis[m]->col(n);
      col++;
    }
  }
  return V;
}

/**
 * @brief transform the circuit parameters using  orthogonal basis calculated by
 * arnoldi process.
 *
 * @param cpi initial circuit parameters.
 * @param cpt transformed curcuit parameters.
 * @param k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 */
void ArnoldiROM::arnoldiTransfer(const CircuitParamInit& cpi,
                                 CircuitParamTrans& cpt, int k) {
  auto size = cpi.C.rows();
  MatrixXd A;
  VectorXd u(size);
  u.setOnes();
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    sum += cpi.C(i, i);
  }
  u = u / sum;
  A = cpi.G.inverse() * cpi.C;
  auto v = genArnoldiBasis(A, u, k);
  if (v) {
    auto arnoldi_basis = v.value();
    cpt.G_rom = arnoldi_basis.transpose() * cpi.G * arnoldi_basis;
    cpt.C_rom = arnoldi_basis.transpose() * cpi.C * arnoldi_basis;
    cpt.B_rom = arnoldi_basis.transpose() * cpi.B;
    cpt.L_rom = arnoldi_basis.transpose() * cpi.L;
  }
}
/**
 * @brief use arnoldi process to calculate the orthogonal basis of Krylov
 * subspace.
 *
 * @param G the conductance matrix.
 * @param C the capacitance matrix.
 * @param k k number of krylov space steps,the number of orthonormal vectors is
 * k+1.
 * @return Eigen::MatrixXd
 */
std::optional<MatrixXd> ArnoldiROM::orthogonalBasis(const MatrixXd& G,
                                                    const MatrixXd& C,
                                                    const VectorXd& B, int k) {
  // diag decompose. C.inv * G = W * diag * W_inv
  MatrixXd cap_inv_dot_conductance_matrix = C.inverse() * G;
  EigenSolver<MatrixXd> es(cap_inv_dot_conductance_matrix);
  MatrixXd diag = es.pseudoEigenvalueMatrix();
  const auto& W = es.pseudoEigenvectors();
  auto W_inv = W.inverse();

  // A is -G_inv * C
  MatrixXd A = -W * diag.inverse() * W_inv;
  DVERBOSE_VLOG(1) << "A matrix\n" << A;

  DVERBOSE_VLOG(1) << "B matrix\n" << B;
  // b is G_inv * B
  MatrixXd b = -A * C.inverse() * B;

  DVERBOSE_VLOG(1) << "b matrix\n" << b;

  auto v = genArnoldiBasis(A, b, k);

  DVERBOSE_VLOG_IF(1, v) << "v matrix" << *v;
  return v;
}

double WaveformApproximation::calVoltageThreshold(MatrixXd& T, MatrixXd& CU,
                                                  double Ceff, int num) {
  int size = T.cols();
  double time_step = (T(0, size - 1) - T(0, 0)) / num;
  double t0 = T(0, 0);
  MatrixXd current(1, num + 1);
  current.setZero();
  current(0, 0) = CU(0, 0);
  double voltage = 0;
  double time = 0;
  for (int i = 1; i < num; i++) {
    // std::cout << t0 + i * time_step << std::endl;
    current(0, i) = EigenMatrixUtility::interplot(t0 + i * time_step, T, CU);

    voltage = voltage + ((current(0, i - 1) + current(0, i)) / 2) * time_step;
  }
  double voltage0 = voltage / Ceff;
  voltage = 0;
  for (int i = 1; i < num; i++) {
    current(0, i) = EigenMatrixUtility::interplot(t0 + i * time_step, T, CU);

    voltage = voltage + ((current(0, i - 1) + current(0, i)) / 2) * time_step;
    double voltage1 = voltage / Ceff;
    if (voltage1 >= voltage0 / 2) {
      std::cout << "voltage0: " << voltage0 << std::endl;
      std::cout << "voltage1: " << voltage1 << std::endl;
      time = i * time_step;
      break;
    }
  }
  MatrixXd curr = current * 1e3;

  return time;
}

double WaveformApproximation::calCeff1(PiModel& pi_model, double t50,
                                       double tr) {
  double t20 = t50 - tr / 2;

  double temp1 =
      (pow((pi_model.R * pi_model.C_far), 2) / (t20 * (t50 - 0.5 * t20))) *
      pow(E, (t20 - t50) / (pi_model.R * pi_model.C_far)) *
      (1 - pow(E, -t20 / (pi_model.R * pi_model.C_far)));
  double temp2 =
      1 - ((pi_model.R * pi_model.C_far) / (t50 - 0.5 * t20)) + temp1;
  double Ceff = pi_model.C_near + pi_model.C_far * temp2;
  return Ceff;
}

double WaveformApproximation::calCeff2(PiModel& pi_model, double t50) {
  double temp1 = 1 - pow(E, -t50 / (pi_model.R * pi_model.C_far));
  double temp2 = 1 - ((pi_model.R * pi_model.C_far) / t50) * temp1;
  double Ceff = pi_model.C_near + pi_model.C_far * temp2;
  return Ceff;
}

PiModel WaveformApproximation::reduceRCTreeToPIModel(
    RcTree& rc_tree, double load_nodes_pin_cap_sum) {
  RctNode* root = rc_tree.get_root();
  LaplaceMoments* root_moments = calMomentsByDFS(root);
  double y1 = root_moments->y1;
  double y2 = root_moments->y2;
  double y3 = root_moments->y3;
  double C1 = pow(y2, 2) / y3;
  double C2 = y1 - pow(y2, 2) / y3;
  double R = -pow(y3, 2) / pow(y2, 3);
  PiModel pi_model{0, 0, 0};
  pi_model.C_near = C2;
  pi_model.R = R;
  C1 += load_nodes_pin_cap_sum;
  pi_model.C_far = C1;
  return pi_model;
}

LaplaceMoments* WaveformApproximation::calMomentsByDFS(RctNode* the_node) {
  the_node->set_is_visited(true);
  LaplaceMoments* the_node_moments = the_node->get_moments();
  if (the_node->get_fanout().size() == 1 && the_node->get_obj()) {
    the_node_moments->y1 = the_node->get_cap();
    the_node_moments->y2 = 0;
    the_node_moments->y3 = 0;
    return the_node_moments;
  }
  for (auto* fanout_edge : the_node->get_fanout()) {
    if (!fanout_edge->get_to().isVisited()) {
      // LaplaceMoments propagate_Y = propagateY(fanout_edge);
      // the_node_moments->y1 += propagate_Y.y1;
      // the_node_moments->y2 += propagate_Y.y2;
      // the_node_moments->y3 += propagate_Y.y3;
    }
  }
  the_node_moments->y1 += the_node->get_cap();
  return the_node_moments;
}

LaplaceMoments WaveformApproximation::propagateY(RctEdge* the_edge) {
  double R = the_edge->get_res();
  LaplaceMoments* load_moments = calMomentsByDFS(&(the_edge->get_to()));
  RctNode* from_node = &(the_edge->get_from());
  double from_y1 = the_edge->get_to().get_moments()->y1;
  double load_y1 = load_moments->y1;
  double load_y2 = load_moments->y2;
  double load_y3 = load_moments->y3;
  from_y1 = load_y1;
  double from_y2 = load_y2 - R * pow(from_y1, 2);
  double from_y3 =
      load_y3 - 2 * R * load_y1 * load_y2 + pow(R, 2) * pow(load_y1, 3);
  from_node->get_moments()->y1 += from_y1;
  from_node->get_moments()->y2 += from_y2;
  from_node->get_moments()->y3 += from_y3;
  LaplaceMoments* from_node_moments = from_node->get_moments();
  return *from_node_moments;
}

double WaveformApproximation::calInputWaveformThresholdByCeff(
    RcTree& rc_tree, double load_nodes_pin_cap_sum, Eigen::MatrixXd& current,
    Eigen::MatrixXd& time, int step_num, TransType trans_type,
    double input_slew, LibertyArc* lib_arc) {
  WaveformApproximation waveform;
  PiModel pi_model =
      waveform.reduceRCTreeToPIModel(rc_tree, load_nodes_pin_cap_sum);
  double cap = pi_model.C_near + pi_model.C_far;
  int iter_num = 500;
  double Ceff = 0;
  double t50 = 0;
  for (int i = 0; i < iter_num; i++) {
    Ceff = cap * 1e12;
    double output_slew = lib_arc->getSlew(trans_type, input_slew, Ceff);
    t50 = calVoltageThreshold(time, current, Ceff, step_num);
    double tr = output_slew * 1e-9;
    Ceff = calCeff1(pi_model, t50, tr);
    if (abs(Ceff * 1e15 - cap * 1e15) < 0.001) {
      break;
    } else {
      cap = Ceff;
    }
  }
  return t50;
}
double WaveformApproximation::calInputWaveformThresholdByCtotal(
    double C_total, Eigen::MatrixXd& current, Eigen::MatrixXd& time,
    int input_step_num) {
  double t50 = calVoltageThreshold(time, current, C_total, input_step_num);
  return t50;
}
void WaveformApproximation::calOutputWaveformThreshold(
    Eigen::MatrixXd& G, Eigen::MatrixXd& C, int iter_num, double tolerence,
    Eigen::MatrixXd& time, int step_num, Eigen::MatrixXd& current,
    std::vector<RctNode*>& load_nodes,
    std::unordered_map<RctNode*, unsigned>& nodes_id, TransType trans_type) {
  int time_size = time.cols();
  int G_size = G.cols();
  double step = (time(0, time_size - 1) - time(0, 0)) / step_num;
  MatrixXd AX = G / 2 + C / step;
  MatrixXd GX = C / step - G / 2;
  MatrixXd time_step(1, step_num + 1);
  MatrixXd cu_interp(1, step_num + 1);

  time_step(0, 0) = time(0, 0);
  cu_interp(0, 0) = current(0, 0);
  for (int i = 1; i <= step_num; ++i) {
    time_step(0, i) = time_step(0, i - 1) + step;
    double cu_inter =
        EigenMatrixUtility::interplot(time_step(0, i), time, current);
    cu_interp(0, i) = cu_inter;
  }

  std::cout << "time_step = " << time_step(0, 0) << std::endl;
  std::cout << "time_step = " << time_step(0, 600) << std::endl;
  std::cout << "time_step = " << time_step(0, 800) << std::endl;

  std::cout << "cu_interp = " << cu_interp(0, 0) << std::endl;
  std::cout << "cu_interp = " << cu_interp(0, 600) << std::endl;
  std::cout << "cu_interp = " << cu_interp(0, 800) << std::endl;

  std::map<RctNode*, std::vector<double>> loads_voltages =
      saveLoadsWaveformVoltages(G_size, AX, GX, iter_num, tolerence, step_num,
                                cu_interp, load_nodes, nodes_id);
  double slew_coefficient = 0.2;
  calOutputWaveformThresholdAndSlew(step, loads_voltages, slew_coefficient,
                                    trans_type);
}
std::map<RctNode*, std::vector<double>>
WaveformApproximation::saveLoadsWaveformVoltages(
    int G_size, Eigen::MatrixXd& AX, Eigen::MatrixXd& GX, int iter_num,
    double tolerence, int step_num, Eigen::MatrixXd& cu_interp,
    std::vector<RctNode*>& load_nodes,
    std::unordered_map<RctNode*, unsigned>& nodes_id) {
  VectorXd x0(G_size);
  x0.setZero();
  VectorXd b1(G_size);
  b1.setZero();

  VectorXd x_gauss(G_size, 1);
  x_gauss.setZero();
  std::map<RctNode*, std::vector<double>> loads_voltages;
  for (int i = 0; i < step_num; ++i) {
    double curr = cu_interp(0, i) + cu_interp(0, i + 1);
    b1.setZero();
    b1(0, 0) = curr / 2;
    VectorXd b = GX * x0 + b1;
    std::cout << "AX = " << AX << std::endl;
    std::cout << "b = " << b << std::endl;
    x_gauss = EigenMatrixUtility::gaussSeidel(AX, b, iter_num, tolerence);
    for (auto* load_node : load_nodes) {
      int load_noad_id = nodes_id[load_node];
      loads_voltages[load_node].push_back(x_gauss(load_noad_id, 0));
    }
    x0 = x_gauss;
  }
  return loads_voltages;
}
void WaveformApproximation::calOutputWaveformThresholdAndSlew(
    double step, std::map<RctNode*, std::vector<double>>& load_voltages,
    double slew_coefficient, TransType trans_type) {
  for (auto& load_voltage : load_voltages) {
    RctNode* load = load_voltage.first;
    std::vector<double> voltages = load_voltage.second;
    int size = voltages.size();
    double former_voltage{0}, later_voltage{0}, vdd{0}, former_time{0},
        middle_time{0}, later_time{0};
    if (trans_type == TransType::kRise) {
      vdd = voltages[size - 1];
    } else {
      vdd = voltages[0];
    }

    double threshold_voltage = vdd / 2;

    if (trans_type == TransType::kRise && slew_coefficient < 0.5) {
      former_voltage = slew_coefficient * vdd;
      later_voltage = (1 - slew_coefficient) * vdd;
    } else if (trans_type == TransType::kRise && slew_coefficient > 0.5) {
      former_voltage = (1 - slew_coefficient) * vdd;
      later_voltage = slew_coefficient * vdd;
    } else if (trans_type == TransType::kFall && slew_coefficient > 0.5) {
      former_voltage = slew_coefficient * vdd;
      later_voltage = (1 - slew_coefficient) * vdd;
    } else if (trans_type == TransType::kFall && slew_coefficient < 0.5) {
      former_voltage = (1 - slew_coefficient) * vdd;
      later_voltage = slew_coefficient * vdd;
    }

    for (int i = 0; i < size; i++) {
      if (trans_type == TransType::kRise) {
        if (voltages[i] < former_voltage && voltages[i + 1] > former_voltage) {
          former_time = (i + 1) * step;

        } else if (voltages[i] < threshold_voltage &&
                   voltages[i + 1] > threshold_voltage) {
          middle_time = (i + 1) * step;
          std::cout << "i= " << i << std::endl;
          std::cout << "voltage = " << voltages[i + 1] << std::endl;
        } else if (voltages[i] < later_voltage &&
                   voltages[i + 1] > later_voltage) {
          later_time = (i + 1) * step;
        }
      } else {
        if (voltages[i] > former_voltage && voltages[i + 1] < former_voltage) {
          former_time = (i + 1) * step;

        } else if (voltages[i] > threshold_voltage &&
                   voltages[i + 1] < threshold_voltage) {
          middle_time = (i + 1) * step;
        }

        else if (voltages[i] > later_voltage &&
                 voltages[i + 1] < later_voltage) {
          later_time = (i + 1) * step;
        }
      }
    }
    _load_nodes_delay[load] = middle_time;
    _load_nodes_slew[load] = later_time - former_time;
  }
}

}  // namespace ista