/**
 * @file IRRustC.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C wrapper for iIR Rust.
 * @version 0.1
 * @date 2024-04-11
 */
#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <vector>

/**
 * Rust vec to C vec
 */
typedef struct RustVec {
  void *data;
  uintptr_t len;
  uintptr_t cap;
  uintptr_t type_size;
} RustVec;

/**
 * Build RC matrix and current vector data.
 */
struct RustVec build_matrix_from_raw_data(const char *c_inst_power_path,
                                          const char *c_power_net_spef);

namespace iir {

void BuildMatrixFromRawData(const char *c_inst_power_path,
                            const char *c_power_net_spef);

}