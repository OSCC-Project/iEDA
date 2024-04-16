/**
 * @file IRRustC.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C wrapper for iIR Rust.
 * @version 0.1
 * @date 2024-04-11
 */
#pragma once

#include "RustCommon.hh"

extern "C" {

typedef struct RustMatrix {
  // val at (row,col)
  double data;
  uintptr_t row;
  uintptr_t col;
} RustMatrix;

typedef struct RustNetConductanceData {
  char *net_name;
  uintptr_t node_num;
  struct RustVec g_matrix_vec;
  // for free Rust ir net memory, record the ptr. 
  const void *ir_net_raw_ptr;
} RustNetConductanceData;

const void *read_spef(const char *c_power_net_spef);

struct RustNetConductanceData build_one_net_conductance_matrix_data(
    const void *c_rc_data, const char *c_net_name);

/**
 * Build RC matrix and current vector data.
 */
struct RustVec build_matrix_from_raw_data(const char *c_inst_power_path,
                                          const char *c_power_net_spef);
}

namespace iir {

void BuildMatrixFromRawData(const char *c_inst_power_path,
                            const char *c_power_net_spef);

}