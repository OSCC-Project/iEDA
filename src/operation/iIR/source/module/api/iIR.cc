/**
 * @file CalcIRDrop.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief
 * @version 0.1
 * @date 2023-08-18
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <string_view>

#include "iIR.hh"
#include "matrix/IRMatrix.hh"
#include "ir-solver/IRSolver.hh"

namespace iir {

/**
 * @brief read spef file.
 *
 * @param spef_file_path
 * @return 
 */
unsigned iIR::readSpef(std::string_view spef_file_path) {
  _rc_data = read_spef(spef_file_path.data());
  return 1;
};

/**
 * @brief read instance power db file to build current vector.
 * 
 * @return unsigned 
 */
unsigned iIR::readInstancePowerDB(std::string_view instance_power_file_path) {
  return 1;
}

/**
 * @brief solve the power net IR drop.
 *
 */
unsigned iIR::solveIRDrop(const char* net_name) {
  auto one_net_matrix_data =
      build_one_net_conductance_matrix_data(_rc_data, net_name);

  IRMatrix ir_matrix;
  auto G_matrix = ir_matrix.buildConductanceMatrix(one_net_matrix_data);

  Eigen::VectorXd J_vector(one_net_matrix_data.node_num);
  J_vector.setZero();
  // TODO(to taosimin), get instance power and calculate the current.

  IRSolver ir_solver;
  auto grid_voltages = ir_solver(G_matrix, J_vector);

  return 1;
}

}  // namespace iir