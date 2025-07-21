/**
 * @file IRRustC.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C wrapper for iIR Rust.
 * @version 0.1
 * @date 2024-04-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "IRRustC.hh"

namespace iir {

void BuildMatrixFromRawData(const char *c_inst_power_path,
                            const char *c_power_net_spef) {
  // auto matrix_data = build_matrix_from_raw_data(c_inst_power_path, c_power_net_spef);
  // fix warning: variable 'matrix_data' set but not used [-Wunused-but-set-variable]
  build_matrix_from_raw_data(c_inst_power_path, c_power_net_spef);

}

}  // namespace iir
