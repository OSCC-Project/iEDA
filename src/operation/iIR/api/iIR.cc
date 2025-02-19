/**
 * @file iIR.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief The top interface of the iIR tools.
 * @version 0.1
 * @date 2023-08-18
 *
 */
#include "iIR.hh"

#include <string_view>

#include "ir-solver/IRSolver.hh"
#include "log/Log.hh"
#include "matrix/IRMatrix.hh"

namespace iir {

/**
 * @brief init IR.
 *
 * @return unsigned
 */
unsigned iIR::init() {
  init_iir();
  return 1;
}

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
  _power_data = read_inst_pwr_csv(instance_power_file_path.data());
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

  auto* current_rust_map =
      build_one_net_instance_current_vector(_power_data, _rc_data, net_name);
  auto J_vector = ir_matrix.buildCurrentVector(current_rust_map,
                                               one_net_matrix_data.node_num);

  IRSolver ir_solver;
  auto grid_voltages = ir_solver(G_matrix, J_vector);

  auto instance_node_ids = get_instance_node_ids(_rc_data, net_name);
  uintptr_t* instance_id;
  FOREACH_VEC_ELEM(&instance_node_ids, uintptr_t, instance_id) {
    LOG_INFO << "instance id " << *instance_id << " ir drop "
             << grid_voltages[*instance_id];
  }

  return 1;
}

}  // namespace iir