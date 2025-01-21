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
 * @file LibArc.cuh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement lookup table.
 * @version 0.1
 * @date 2025-01-03
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <climits>

#include "propagation.cuh"

namespace ista {
/**
 * @brief The struct of Axis_Region, used to store the return value of
 * func:get_axis_region.
 *
 */
struct Axis_Region {
  double x1;
  double x2;
  unsigned val_index;
};

/**
 * @brief find value of LibTableGPU according to slew and
 * constrain_slew_or_load.
 * @param lib_table_gpu
 * @param slew
 * @param constrain_slew_or_load
 */
__device__ double find_value(LibTableGPU& lib_table_gpu, double slew,
                             double constrain_slew_or_load);

/**
 * @brief build gpu LibArcGPU(lib_data_gpu._arcs_gpu) according to cpu
 * LibArcGPU(lib_arcs_cpu).
 * @param lib_data_gpu The struct of LibDataGPU.
 * @param lib_arcs_cpu The vector of LibArcGPU.
 */
void build_lib_data_gpu(LibDataGPU& lib_data_gpu,
                        std::vector<LibArcGPU*> lib_arcs_cpu);

/**
 * @brief for test.
 */
__global__ void kernel_find_value(LibDataGPU& lib_data_gpu, double slew,
                                  double constrain_slew_or_load,
                                  double* d_value);

/**
 * @brief for test.
 */
double find_value(LibDataGPU& lib_data_gpu, double slew,
                  double constrain_slew_or_load);

}  // namespace ista