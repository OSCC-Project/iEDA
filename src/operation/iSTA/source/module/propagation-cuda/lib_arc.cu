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
 * @file LibArc.cu
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement lookup table.
 * @version 0.1
 * @date 2025-01-09
 */

#include <stdio.h>

#include <cassert>
#include <iomanip>
#include <iostream>

#include "gpu/cuda_common.cuh"
#include "lib_arc.cuh"

namespace ista {

__device__ constexpr float float_precision = 1e-15;

__device__ inline bool is_float_equal(float data1, float data2,
                                float epsilon = float_precision) {
  return fabs(data1 - data2) < epsilon;
}

__device__ constexpr float DOUBLE_MAX = 1.7976931348623157e+308;
__device__ constexpr float DOUBLE_MIN = -1.7976931348623157e+308;

/**
 * @brief The one dimension interpolation.
 *
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @param x
 * @return float
 */
__device__ inline float linear_interpolate(float x1, float x2, float y1, float y2,
                                     float x) {
  assert(!is_float_equal(x1, x2));

  if (x >= DOUBLE_MAX || x <= DOUBLE_MIN) {
    return x;
  }

  float slope = (y2 - y1) / (x2 - x1);
  float ret_val;

  if (x < x1) {
    ret_val = y1 - (x1 - x) * slope;  // Extrapolation.
  } else if (x > x2) {
    ret_val = y2 + (x - x2) * slope;  // Extrapolation.
  } else if (is_float_equal(x, x1)) {
    ret_val = y1;  // Boundary case.
  } else if (is_float_equal(x, x2)) {
    ret_val = y2;  // Boundary case.
  } else {
    ret_val = y1 + (x - x1) * slope;  // Interpolation.
  }

  return ret_val;
}

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
 * @return float
 */
__device__ inline float bilinear_interpolation(float q11, float q12, float q21,
                                         float q22, float x1, float x2,
                                         float y1, float y2, float x,
                                         float y) {
  const float x2x1 = x2 - x1;
  const float y2y1 = y2 - y1;
  const float x2x = x2 - x;
  const float y2y = y2 - y;
  const float yy1 = y - y1;
  const float xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

/**
 * @brief get x axis size of Lib_Table_GPU.
 * @param lib_table_gpu
 */
__device__ inline float get_x_axis_size(Lib_Table_GPU& lib_table_gpu) {
  return lib_table_gpu._num_x;
}

/**
 * @brief get x val size of Lib_Table_GPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ inline float get_x_axis_val(Lib_Table_GPU& lib_table_gpu, unsigned index) {
  return lib_table_gpu._x[index];
}

/**
 * @brief get y axis size of Lib_Table_GPU.
 * @param lib_table_gpu
 */
__device__ inline float get_y_axis_size(Lib_Table_GPU& lib_table_gpu) {
  return lib_table_gpu._num_y;
}

/**
 * @brief get y val size of Lib_Table_GPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ inline float get_y_axis_val(Lib_Table_GPU& lib_table_gpu, unsigned index) {
  return lib_table_gpu._y[index];
}

/**
 * @brief get table value of Lib_Table_GPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ inline float get_table_value(Lib_Table_GPU& lib_table_gpu, unsigned index) {
  if (index >= lib_table_gpu._num_values) {
    CUDA_LOG_ERROR("index %u beyond table value size %u", index,
           lib_table_gpu._num_values);
    return -1.0;
  }
  return lib_table_gpu._values[index];
}

/**
 * @brief check val of Lib_Table_GPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param val
 */
__device__ inline unsigned check_val(Lib_Table_GPU& lib_table_gpu, int axis_index,
                              float val) {
  unsigned num_val = 0;
  float min_val = 0;
  float max_val = 0;
  if (axis_index == 0) {
    num_val = get_x_axis_size(lib_table_gpu);
    min_val = get_x_axis_val(lib_table_gpu, 0);
    max_val = get_x_axis_val(lib_table_gpu, num_val - 1);
  } else if (axis_index == 1) {
    num_val = get_y_axis_size(lib_table_gpu);
    min_val = get_y_axis_val(lib_table_gpu, 0);
    max_val = get_y_axis_val(lib_table_gpu, num_val - 1);
  }

  if ((val < min_val) || (val > max_val)) {
    // CUDA_LOG_WARNING(
    //     "val outside table ranges: val = %f; min_val = %f; max_val "
    //     "= %f",
    //     val, min_val, max_val);
  }
  return num_val;
}

/**
 * @brief get val's axis region of Lib_Table_GPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param num_val
 * @param val
 */
__device__ Axis_Region get_axis_region(Lib_Table_GPU& lib_table_gpu,
                                       int axis_index, unsigned int num_val,
                                       float val) {
  float x2 = 0.0;
  unsigned int val_index = 0;
  float x1;
  if (axis_index == 0) {
    for (; val_index < num_val; val_index++) {
      x2 = get_x_axis_val(lib_table_gpu, val_index);
      if (x2 > val) {
        break;
      }
    }

    if (val_index == num_val) {
      val_index = num_val - 2;
    } else if (val_index) {
      --val_index;
    } else {
      x2 = get_x_axis_val(lib_table_gpu, 1);
    }
    x1 = get_x_axis_val(lib_table_gpu, val_index);
  } else if (axis_index == 1) {
    for (; val_index < num_val; val_index++) {
      x2 = get_y_axis_val(lib_table_gpu, val_index);
      if (x2 > val) {
        break;
      }
    }

    if (val_index == num_val) {
      val_index = num_val - 2;
    } else if (val_index) {
      --val_index;
    } else {
      x2 = get_y_axis_val(lib_table_gpu, 1);
    }
    x1 = get_y_axis_val(lib_table_gpu, val_index);
  }

  return Axis_Region{x1, x2, val_index};
}

/**
 * @brief find value according to slew and constrain_slew_or_load.
 * @param slew The slew value.
 * @param constrain_slew_or_load The constrain_slew_or_load value.
 * @return The value.
 */
__device__ float find_value(Lib_Table_GPU& lib_table_gpu, float slew,
                             float constrain_slew_or_load) {
  // ??? not sure (_type == UINT_MAX) can work as (!table_template)
  if (lib_table_gpu._type == UINT_MAX) {
    CUDA_LOG_DEBUG("the table type is not set");
    return 0.0;
  }

  float val1;
  float val2;
  if (lib_table_gpu._type == 0 || lib_table_gpu._type == 2) {
    val1 = slew;
    val2 = constrain_slew_or_load;
  } else if (lib_table_gpu._type == 1 || lib_table_gpu._type == 3) {
    val1 = constrain_slew_or_load;
    val2 = slew;
  } else {
    // ??? not sure (_type == UINT_MAX) can work as (!table_template)
    CUDA_LOG_ERROR("lut table: invalid delay lut template variable");
    return 0.0;
  }

  if (lib_table_gpu._num_y == 0) {
    // only one axis(x axis.)
    auto num_val1 = check_val(lib_table_gpu, 0, val1);
    auto [x1, x2, val1_index] =
        get_axis_region(lib_table_gpu, 0, num_val1, val1);
    unsigned int x1_table_val = get_table_value(lib_table_gpu, val1_index);
    unsigned int x2_table_val = get_table_value(lib_table_gpu, val1_index + 1);

    auto result = linear_interpolate(x1, x2, x1_table_val, x2_table_val, val1);
    return result;
  } else {
    // two axis(x and y axis.)
    auto num_val1 = check_val(lib_table_gpu, 0, val1);
    auto num_val2 = check_val(lib_table_gpu, 1, val2);

    auto [x1, x2, val1_index] =
        get_axis_region(lib_table_gpu, 0, num_val1, val1);
    auto [y1, y2, val2_index] =
        get_axis_region(lib_table_gpu, 1, num_val2, val2);

    // now do the table lookup
    unsigned int index = num_val2 * val1_index + val2_index;
    const auto q11 = get_table_value(lib_table_gpu, index);

    index = num_val2 * (val1_index + 1) + val2_index;
    const auto q21 = get_table_value(lib_table_gpu, index);

    index = num_val2 * val1_index + (val2_index + 1);
    const auto q12 = get_table_value(lib_table_gpu, index);

    index = num_val2 * (val1_index + 1) + (val2_index + 1);
    const auto q22 = get_table_value(lib_table_gpu, index);

    auto result =
        bilinear_interpolation(q11, q12, q21, q22, x1, x2, y1, y2, val1, val2);

    // CUDA_LOG_INFO("x1 %f x2 %f y1 %f y2 %f val1 %f val2 %f result %f", x1, x2, y1, y2, val1, val2, result);
    return result;
  }
}

/**
 * @brief build gpu Lib_Arc_GPU(lib_data_gpu._arcs_gpu) according to cpu
 * Lib_Arc_GPU(lib_arcs_cpu).
 * @param lib_data_gpu The struct of Lib_Data_GPU.
 * @param lib_arcs_cpu The vector of Lib_Arc_GPU.
 */
void build_lib_data_gpu(Lib_Data_GPU& lib_data_gpu,
                        std::vector<Lib_Table_GPU>& lib_tables_gpu,
                        std::vector<Lib_Table_GPU*>& lib_gpu_table_ptrs,
                        std::vector<Lib_Arc_GPU>& lib_arcs_cpu) {
  lib_data_gpu._num_arcs = lib_arcs_cpu.size();

  cudaMalloc(&(lib_data_gpu._arcs_gpu),
             lib_data_gpu._num_arcs * sizeof(Lib_Arc_GPU));

  cudaStream_t stream1 = nullptr;
  cudaStreamCreate(&stream1);

  for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
    Lib_Arc_GPU& cpu_arc = lib_arcs_cpu[i];

    Lib_Table_GPU* d_tables;
    CUDA_CHECK(
        cudaMalloc(&(d_tables), cpu_arc._num_table * sizeof(Lib_Table_GPU)));
    CUDA_CHECK(cudaMemcpy(d_tables, cpu_arc._table,
                          cpu_arc._num_table * sizeof(Lib_Table_GPU),
                          cudaMemcpyHostToDevice));
    lib_gpu_table_ptrs.push_back(d_tables);

    for (unsigned j = 0; j < cpu_arc._num_table; ++j) {
      Lib_Table_GPU& cpu_table = cpu_arc._table[j];
      Lib_Table_GPU* gpu_table = &d_tables[j];

      Lib_Table_GPU one_table;

      CUDA_CHECK(
          cudaMallocAsync(&(one_table._x), cpu_table._num_x * sizeof(float), stream1));
      CUDA_CHECK(
          cudaMallocAsync(&(one_table._y), cpu_table._num_y * sizeof(float), stream1));
      CUDA_CHECK(cudaMallocAsync(
          &(one_table._values), cpu_table._num_values * sizeof(float), stream1));
      CUDA_CHECK(cudaStreamSynchronize(stream1));

      // assign gpu table data.
      CUDA_CHECK(cudaMemcpyAsync(one_table._x, cpu_table._x,
                                 cpu_table._num_x * sizeof(float),
                                 cudaMemcpyHostToDevice, stream1));
      CUDA_CHECK(cudaMemcpyAsync(one_table._y, cpu_table._y,
                                 cpu_table._num_y * sizeof(float),
                                 cudaMemcpyHostToDevice, stream1));
      CUDA_CHECK(cudaMemcpyAsync(one_table._values, cpu_table._values,
                                 cpu_table._num_values * sizeof(float),
                                 cudaMemcpyHostToDevice, stream1));
      CUDA_CHECK(cudaStreamSynchronize(stream1));

      one_table._num_x = cpu_table._num_x;
      one_table._num_y = cpu_table._num_y;
      one_table._type = cpu_table._type;
      one_table._num_values = cpu_table._num_values;

      CUDA_CHECK(cudaMemcpyAsync(gpu_table, &one_table,
                                 sizeof(Lib_Table_GPU), cudaMemcpyHostToDevice,
                                 stream1));
      CUDA_CHECK(cudaStreamSynchronize(stream1));

      lib_tables_gpu.emplace_back(std::move(one_table));
    }


    // assign gpu arc data.
    Lib_Arc_GPU* gpu_arc = &lib_data_gpu._arcs_gpu[i];

    CUDA_CHECK(cudaMemcpyAsync(&(gpu_arc->_table), &d_tables,
                               sizeof(Lib_Table_GPU*), cudaMemcpyHostToDevice,
                               stream1));
    CUDA_CHECK(cudaMemcpyAsync(&(gpu_arc->_num_table), &cpu_arc._num_table,
                               sizeof(unsigned), cudaMemcpyHostToDevice,
                               stream1));
    CUDA_CHECK(cudaMemcpyAsync(&(gpu_arc->_line_no), &cpu_arc._line_no,
                               sizeof(unsigned), cudaMemcpyHostToDevice,
                               stream1));
    CUDA_CHECK(cudaMemcpyAsync(&(gpu_arc->_cap_unit), &cpu_arc._cap_unit,
                               sizeof(Lib_Cap_unit), cudaMemcpyHostToDevice,
                               stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
  }
  cudaStreamDestroy(stream1);
}

/**
 * @brief free gpu data memory.
 * 
 * @param lib_data_gpu 
 * @param lib_tables_gpu 
 */
void free_lib_data_gpu(Lib_Data_GPU& lib_data_gpu,
                       std::vector<Lib_Table_GPU>& lib_tables_gpu,
                       std::vector<Lib_Table_GPU*>& lib_gpu_table_ptrs) { 

  for (auto& lib_table : lib_tables_gpu) {
    cudaFree(lib_table._x);
    cudaFree(lib_table._y);
    cudaFree(lib_table._values);
  }

  for (auto* lib_table_ptr : lib_gpu_table_ptrs) {
    cudaFree(lib_table_ptr);
  }
  
  if (lib_data_gpu._arcs_gpu) {
    cudaFree(lib_data_gpu._arcs_gpu);
  }
  
}

/**
 * @brief print first arc's first table of lib_data_gpu for debug.
 */
void print_lib_data_gpu_first_arc_first_table(Lib_Data_GPU& lib_data_gpu) {
  // gpu->cpu transfer
  Lib_Arc_GPU* h_arcs;
  cudaMallocHost(&h_arcs, sizeof(Lib_Arc_GPU) * lib_data_gpu._num_arcs);

  cudaMemcpy(h_arcs, lib_data_gpu._arcs_gpu,
             sizeof(Lib_Arc_GPU) * lib_data_gpu._num_arcs,
             cudaMemcpyDeviceToHost);

  Lib_Table_GPU* h_table;
  cudaMallocHost(&h_table, sizeof(Lib_Table_GPU) * h_arcs[0]._num_table);
  cudaMemcpy(h_table, h_arcs[0]._table,
             sizeof(Lib_Table_GPU) * h_arcs[0]._num_table,
             cudaMemcpyDeviceToHost);

  Lib_Table_GPU first_table = h_table[0];

  std::cout << "First table values:" << std::endl;
  std::cout << "Num X: " << first_table._num_x << std::endl;
  std::cout << "Num Y: " << first_table._num_y << std::endl;
  std::cout << "Num Values: " << first_table._num_values << std::endl;

  float* h_x;
  float* h_y;
  float* h_values;

  cudaMallocHost(&h_x, sizeof(float) * first_table._num_x);
  cudaMallocHost(&h_y, sizeof(float) * first_table._num_y);
  cudaMallocHost(&h_values, sizeof(float) * first_table._num_values);

  cudaMemcpy(h_x, first_table._x, sizeof(float) * first_table._num_x,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_y, first_table._y, sizeof(float) * first_table._num_y,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_values, first_table._values,
             sizeof(float) * first_table._num_values, cudaMemcpyDeviceToHost);

  std::cout << "X: ";
  for (unsigned i = 0; i < first_table._num_x; ++i) {
    std::cout << h_x[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Y: ";
  for (unsigned i = 0; i < first_table._num_y; ++i) {
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Values: ";
  for (unsigned i = 0; i < first_table._num_values; ++i) {
    std::cout << h_values[i] << " ";
  }
  std::cout << std::endl;

  cudaFreeHost(h_arcs);
  cudaFreeHost(h_table);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_values);
}

__global__ void kernel_find_value(Lib_Data_GPU lib_data_gpu, float slew,
                                  float constrain_slew_or_load,
                                  float* d_value) {
  *d_value = find_value(lib_data_gpu._arcs_gpu[0]._table[0], slew,
                        constrain_slew_or_load);
}

float find_value_test(Lib_Data_GPU& lib_data_gpu, float slew,
                  float constrain_slew_or_load) {
  // print first arc's first table of lib_data_gpu for debug.
  print_lib_data_gpu_first_arc_first_table(lib_data_gpu);

  // transfer lib_data_gpu_host(host pointer) to lib_data_gpu_device(device
  // pointer) for launching kernel.
  Lib_Data_GPU lib_data_gpu_host = lib_data_gpu;


  // launch kernel
  float* d_value;
  cudaMalloc((void**)&d_value, sizeof(float));
  kernel_find_value<<<1, 1>>>(lib_data_gpu_host, slew, constrain_slew_or_load,
                              d_value);
  CUDA_CHECK_ERROR();
  cudaDeviceSynchronize();

  float value;
  cudaMemcpy(&value, d_value, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_value);

  return value;
}

}  // namespace ista