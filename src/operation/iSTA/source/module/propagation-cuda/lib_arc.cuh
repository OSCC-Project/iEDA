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
  float x1;
  float x2;
  unsigned val_index;
};

/**
 * @brief find value of Lib_Table_GPU according to slew and
 * constrain_slew_or_load.
 * @param lib_table_gpu
 * @param slew
 * @param constrain_slew_or_load
 */
__device__ float find_value(Lib_Table_GPU& lib_table_gpu, float slew,
                             float constrain_slew_or_load);

/**
 * @brief build gpu Lib_Arc_GPU(lib_data_gpu._arcs_gpu) according to cpu
 * Lib_Arc_GPU(lib_arcs_cpu).
 * @param lib_data_gpu The struct of Lib_Data_GPU.
 * @param lib_arcs_cpu The vector of Lib_Arc_GPU.
 */
void build_lib_data_gpu(Lib_Data_GPU& lib_data_gpu,
                        std::vector<Lib_Table_GPU>& lib_tables_gpu,
                        std::vector<Lib_Table_GPU*>& lib_gpu_table_ptrs,
                        std::vector<Lib_Arc_GPU>& lib_arcs_cpu);

/**
 * @brief free the gpu memory of Lib_Data_GPU.
 * 
 * @param lib_data_gpu 
 * @return * void 
 */
void free_lib_data_gpu(Lib_Data_GPU& lib_data_gpu,
                       std::vector<Lib_Table_GPU>& lib_tables_gpu,
                       std::vector<Lib_Table_GPU*>& lib_gpu_table_ptrs);

/**
 * @brief for test.
 */
float find_value_test(Lib_Data_GPU& lib_data_gpu, float slew,
                  float constrain_slew_or_load);

}  // namespace ista