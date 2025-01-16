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
 * @brief The one dimension interpolation.
 *
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @param x
 * @return double
 */
__device__ double linear_interpolate(double x1, double x2, double y1, double y2,
                                     double x);

/**
 * @brief The two dimension interpolation.
 * // From
 * https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
 * @param q11 x1, y1 value
 * @param q12 x1, y2 value
 * @param q21 x2, y1 value
 * @param q22 x2, y2 value
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @param x
 * @param y
 * @return double
 */
__device__ double bilinear_interpolation(double q11, double q12, double q21,
                                         double q22, double x1, double x2,
                                         double y1, double y2, double x,
                                         double y);

/**
 * @brief get x axis size of LibTableGPU.
 * @param lib_table_gpu
 */
__device__ double get_x_axis_size(LibTableGPU& lib_table_gpu);

/**
 * @brief get x val size of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_x_axis_val(LibTableGPU& lib_table_gpu, unsigned index);

/**
 * @brief get y axis size of LibTableGPU.
 * @param lib_table_gpu
 */
__device__ double get_y_axis_size(LibTableGPU& lib_table_gpu);

/**
 * @brief get y val size of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_y_axis_val(LibTableGPU& lib_table_gpu, unsigned index);

/**
 * @brief get table value of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_table_value(LibTableGPU& lib_table_gpu, unsigned index);

/**
 * @brief check val of LibTableGPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param val
 */
__device__ unsigned check_val(LibTableGPU& lib_table_gpu, int axis_index,
                              double val);

/**
 * @brief The struct of AxisRegion, used to store the return value of
 * func:get_axis_region.
 *
 */
struct AxisRegion {
  double x1;
  double x2;
  unsigned val_index;
};

/**
 * @brief get val's axis region of LibTableGPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param num_val
 * @param val
 */
__device__ AxisRegion get_axis_region(LibTableGPU& lib_table_gpu,
                                      int axis_index, unsigned int num_val,
                                      double val);

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

__global__ void kernel_find_value(LibDataGPU& lib_data_gpu, double slew,
                                  double constrain_slew_or_load,
                                  double* d_value);

double find_value(LibDataGPU& lib_data_gpu, double slew,
                  double constrain_slew_or_load);

}  // namespace ista