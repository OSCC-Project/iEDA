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

namespace iir {

/**
 * @brief
 *
 * @param spef_file_path
 * @return 
 */
unsigned iIR::readSpef(std::string_view spef_file_path) {
  _rc_data = read_spef(spef_file_path.data());
  return 1;
};

/**
 * @brief solve the power net IR drop.
 *
 */
unsigned iIR::solveIRDrop(const char* net_name) {
  auto one_net_matrix_data =
      build_one_net_conductance_matrix_data(_rc_data, net_name);

  IRMatrix ir_matrix;
  auto g_matrix = ir_matrix.buildConductanceMatrix(one_net_matrix_data);

  return 1;
}

}  // namespace iir