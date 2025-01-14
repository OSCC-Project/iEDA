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
#include <vector>

#include "log/Log.hh"

namespace ista {

__device__ double linear_interpolate(double x1, double x2, double y1, double y2,
                                     double x);
__device__ double bilinear_interpolation(double q11, double q12, double q21,
                                         double q22, double x1, double x2,
                                         double y1, double y2, double x,
                                         double y);

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
 * @brief The struct of LibTableGPU.
 *
 */
struct LibTableGPU {
  double* _x;
  double* _y;
  unsigned _num_x = 0;
  unsigned _num_y = 0;
  double* _values;
  unsigned _num_values = 0;
  unsigned _type =
      UINT_MAX;  //!< normal(slew->cap),invert(cap->slew),slew,cap,and so on.

  __device__ double get_x_axis_size() { return _num_x; }
  __device__ double get_x_axis_val(unsigned index) { return _x[index]; }
  __device__ double get_y_axis_size() { return _num_y; }
  __device__ double get_y_axis_val(unsigned index) { return _y[index]; }
  __device__ double get_table_value(unsigned index) {
    if (index >= _num_values) {
      printf("Error: index %u beyond table value size %u\n", index,
             _num_values);
      return -1.0;
    }
    return _values[index];
  }

  __device__ unsigned check_val(int axis_index, double val) {
    unsigned num_val = 0;
    double min_val = 0;
    double max_val = 0;
    if (axis_index == 0) {
      num_val = get_x_axis_size();
      min_val = get_x_axis_val(0);
      max_val = get_x_axis_val(num_val - 1);
    } else if (axis_index == 1) {
      num_val = get_y_axis_size();
      min_val = get_y_axis_val(0);
      max_val = get_y_axis_val(num_val - 1);
    }

    if ((val < min_val) || (val > max_val)) {
      printf(
          "Warning: val outside table ranges: val = %f; min_val = %f; max_val "
          "= %f\n",
          val, min_val, max_val);
    }
    return num_val;
  }

  __device__ AxisRegion get_axis_region(int axis_index, unsigned int num_val,
                                        double val) {
    double x2 = 0.0;
    unsigned int val_index = 0;
    double x1;
    if (axis_index == 0) {
      for (; val_index < num_val; val_index++) {
        x2 = get_x_axis_val(val_index);
        if (x2 > val) {
          break;
        }
      }

      if (val_index == num_val) {
        val_index = num_val - 2;
      } else if (val_index) {
        --val_index;
      } else {
        x2 = get_x_axis_val(1);
      }
      x1 = get_x_axis_val(val_index);
    } else if (axis_index == 1) {
      for (; val_index < num_val; val_index++) {
        x2 = get_y_axis_val(val_index);
        if (x2 > val) {
          break;
        }
      }

      if (val_index == num_val) {
        val_index = num_val - 2;
      } else if (val_index) {
        --val_index;
      } else {
        x2 = get_y_axis_val(1);
      }
      x1 = get_y_axis_val(val_index);
    }

    return AxisRegion{x1, x2, val_index};
  }
  __device__ double find_value(double slew, double constrain_slew_or_load);
};

/**
 * @brief The struct of LibArcGPU.
 *
 */
struct LibArcGPU {
  LibTableGPU* _table;
  unsigned _num_table;  //!< number of tables.(SSTA:12 tables.)
};

/**
 * @brief The struct of LibDataGPU.
 *
 */
struct LibDataGPU {
  LibArcGPU* _arcs_gpu;  //!< points to GPU arc datas.
  unsigned _num_arcs;    //!< GPU arc datas.

  std::vector<LibArcGPU> _arcs;  //!< CPU arc datas.
};

/**
 * @brief build gpu LibArcGPU(lib_data_gpu._arcs_gpu) according to cpu
 * LibArcGPU(lib_arcs_cpu).
 * @param lib_data_gpu The struct of LibDataGPU.
 * @param lib_arcs_cpu The vector of LibArcGPU.
 */
void build_lib_data_gpu(LibDataGPU& lib_data_gpu,
                        std::vector<LibArcGPU*> lib_arcs_cpu);

}  // namespace ista