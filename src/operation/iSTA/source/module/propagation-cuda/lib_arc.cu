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

__device__ constexpr double double_precision = 1e-15;

__device__ bool is_double_equal(double data1, double data2,
                                double epsilon = double_precision) {
  return fabs(data1 - data2) < epsilon;
}

__device__ constexpr double DOUBLE_MAX = 1.7976931348623157e+308;
__device__ constexpr double DOUBLE_MIN = -1.7976931348623157e+308;

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
                                     double x) {
  assert(!is_double_equal(x1, x2));

  if (x >= DOUBLE_MAX || x <= DOUBLE_MIN) {
    return x;
  }

  double slope = (y2 - y1) / (x2 - x1);
  double ret_val;

  if (x < x1) {
    ret_val = y1 - (x1 - x) * slope;  // Extrapolation.
  } else if (x > x2) {
    ret_val = y2 + (x - x2) * slope;  // Extrapolation.
  } else if (is_double_equal(x, x1)) {
    ret_val = y1;  // Boundary case.
  } else if (is_double_equal(x, x2)) {
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
 * @return double
 */
__device__ double bilinear_interpolation(double q11, double q12, double q21,
                                         double q22, double x1, double x2,
                                         double y1, double y2, double x,
                                         double y) {
  const double x2x1 = x2 - x1;
  const double y2y1 = y2 - y1;
  const double x2x = x2 - x;
  const double y2y = y2 - y;
  const double yy1 = y - y1;
  const double xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

/**
 * @brief get x axis size of LibTableGPU.
 * @param lib_table_gpu
 */
__device__ double get_x_axis_size(LibTableGPU& lib_table_gpu) {
  return lib_table_gpu._num_x;
}

/**
 * @brief get x val size of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_x_axis_val(LibTableGPU& lib_table_gpu, unsigned index) {
  return lib_table_gpu._x[index];
}

/**
 * @brief get y axis size of LibTableGPU.
 * @param lib_table_gpu
 */
__device__ double get_y_axis_size(LibTableGPU& lib_table_gpu) {
  return lib_table_gpu._num_y;
}

/**
 * @brief get y val size of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_y_axis_val(LibTableGPU& lib_table_gpu, unsigned index) {
  return lib_table_gpu._y[index];
}

/**
 * @brief get table value of LibTableGPU.
 * @param lib_table_gpu
 * @param index
 */
__device__ double get_table_value(LibTableGPU& lib_table_gpu, unsigned index) {
  if (index >= lib_table_gpu._num_values) {
    printf("Error: index %u beyond table value size %u\n", index,
           lib_table_gpu._num_values);
    return -1.0;
  }
  return lib_table_gpu._values[index];
}

/**
 * @brief check val of LibTableGPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param val
 */
__device__ unsigned check_val(LibTableGPU& lib_table_gpu, int axis_index,
                              double val) {
  unsigned num_val = 0;
  double min_val = 0;
  double max_val = 0;
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
    printf(
        "Warning: val outside table ranges: val = %f; min_val = %f; max_val "
        "= %f\n",
        val, min_val, max_val);
  }
  return num_val;
}

/**
 * @brief get val's axis region of LibTableGPU.
 * @param lib_table_gpu
 * @param axis_index
 * @param num_val
 * @param val
 */
__device__ Axis_Region get_axis_region(LibTableGPU& lib_table_gpu,
                                       int axis_index, unsigned int num_val,
                                       double val) {
  double x2 = 0.0;
  unsigned int val_index = 0;
  double x1;
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
__device__ double find_value(LibTableGPU& lib_table_gpu, double slew,
                             double constrain_slew_or_load) {
  // ??? not sure (_type == UINT_MAX) can work as (!table_template)
  if (lib_table_gpu._type == UINT_MAX) {
    return lib_table_gpu._values[0];
  }

  double val1;
  double val2;
  if (lib_table_gpu._type == 0 || lib_table_gpu._type == 2) {
    val1 = slew;
    val2 = constrain_slew_or_load;
  } else if (lib_table_gpu._type == 1 || lib_table_gpu._type == 3) {
    val1 = constrain_slew_or_load;
    val2 = slew;
  } else {
    // ??? not sure (_type == UINT_MAX) can work as (!table_template)
    printf("Error: lut table: invalid delay lut template variable\n");
    return -1.0;
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

    // LOG_ERROR_IF_EVERY_N(result < 0.0, 100) << "table " << get_file_name() <<
    // " " << get_line_no() << " "
    //                                         << "delay value less zero.";
  }
}

/**
 * @brief build gpu LibArcGPU(lib_data_gpu._arcs_gpu) according to cpu
 * LibArcGPU(lib_arcs_cpu).
 * @param lib_data_gpu The struct of LibDataGPU.
 * @param lib_arcs_cpu The vector of LibArcGPU.
 */
void build_lib_data_gpu(LibDataGPU& lib_data_gpu,
                        std::vector<LibArcGPU*> lib_arcs_cpu_ptr) {
  std::vector<ista::LibArcGPU> lib_arcs_cpu;
  lib_arcs_cpu.reserve(lib_arcs_cpu_ptr.size());
  for (const auto& arc_ptr : lib_arcs_cpu_ptr) {
    if (arc_ptr != nullptr) {
      lib_arcs_cpu.push_back(*arc_ptr);
    }
  }
  // ???  not sure need _arcs as member of LibDataGPU
  lib_data_gpu._arcs = lib_arcs_cpu;

  lib_data_gpu._num_arcs = lib_arcs_cpu_ptr.size();

  cudaMalloc(&(lib_data_gpu._arcs_gpu),
             lib_data_gpu._num_arcs * sizeof(LibArcGPU));

  for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
    LibArcGPU* cpu_arc = lib_arcs_cpu_ptr[i];

    LibTableGPU* d_tables;
    CUDA_CHECK(
        cudaMalloc(&(d_tables), cpu_arc->_num_table * sizeof(LibTableGPU)));
    CUDA_CHECK(cudaMemcpy(d_tables, cpu_arc->_table,
                          cpu_arc->_num_table * sizeof(LibTableGPU),
                          cudaMemcpyHostToDevice));

    for (unsigned j = 0; j < cpu_arc->_num_table; ++j) {
      LibTableGPU& cpu_table = cpu_arc->_table[j];

      double *d_x, *d_y, *d_values;
      CUDA_CHECK(cudaMalloc(&d_x, cpu_table._num_x * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_y, cpu_table._num_y * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_values, cpu_table._num_values * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_x, cpu_table._x,
                            cpu_table._num_x * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_y, cpu_table._y,
                            cpu_table._num_y * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_values, cpu_table._values,
                            cpu_table._num_values * sizeof(double),
                            cudaMemcpyHostToDevice));

      LibTableGPU* gpu_table = &d_tables[j];

      CUDA_CHECK(cudaMemcpy(&(gpu_table->_x), &d_x, sizeof(double*),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(&(gpu_table->_y), &d_y, sizeof(double*),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(&(gpu_table->_values), &d_values, sizeof(double*),
                            cudaMemcpyHostToDevice));

      unsigned num_x = cpu_table._num_x;
      unsigned num_y = cpu_table._num_y;
      unsigned num_values = cpu_table._num_values;

      CUDA_CHECK(cudaMemcpy(&(gpu_table->_num_x), &num_x, sizeof(unsigned),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(&(gpu_table->_num_y), &num_y, sizeof(unsigned),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(&(gpu_table->_num_values), &num_values,
                            sizeof(unsigned), cudaMemcpyHostToDevice));
    }

    LibArcGPU* gpu_arc = &lib_data_gpu._arcs_gpu[i];

    unsigned num_table = cpu_arc->_num_table;
    CUDA_CHECK(cudaMemcpy(&(gpu_arc->_table), &d_tables, sizeof(LibTableGPU*),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&(gpu_arc->_num_table), &num_table, sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }
}

// void build_lib_data_gpu(LibDataGPU& lib_data_gpu,
//                         std::vector<LibArcGPU*> lib_arcs_cpu_ptr) {
//   std::vector<ista::LibArcGPU> lib_arcs_cpu;
//   lib_arcs_cpu.reserve(lib_arcs_cpu_ptr.size());
//   for (const auto& arc_ptr : lib_arcs_cpu_ptr) {
//     if (arc_ptr != nullptr) {
//       lib_arcs_cpu.push_back(*arc_ptr);
//     }
//   }

//   lib_data_gpu._num_arcs = lib_arcs_cpu.size();
//   // ???  not sure need _arcs as member of LibDataGPU
//   lib_data_gpu._arcs = lib_arcs_cpu;
//   // copy cpu data(lib_arcs_cpu) to gpu data(lib_data_gpu._arcs_gpu)
//   CUDA_CHECK(cudaMalloc(&(lib_data_gpu._arcs_gpu),
//                         lib_data_gpu._num_arcs * sizeof(LibArcGPU)));

//   CUDA_CHECK(cudaMemcpy(lib_data_gpu._arcs_gpu, lib_data_gpu._arcs.data(),
//                         lib_data_gpu._num_arcs * sizeof(LibArcGPU),
//                         cudaMemcpyHostToDevice));

//   // copy table data to gpu
//   for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
//     const auto& arc = lib_data_gpu._arcs[i];
//     CUDA_CHECK(cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table),
//                           arc._num_table * sizeof(LibTableGPU)));
//     CUDA_CHECK(cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table, arc._table,
//                           arc._num_table * sizeof(LibTableGPU),
//                           cudaMemcpyHostToDevice));
//     lib_data_gpu._arcs_gpu[i]._num_table = arc._num_table;
//   }

//   // copy table data to gpu,copy x,y and values to gpu
//   for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
//     const auto& arc = lib_data_gpu._arcs[i];
//     for (unsigned j = 0; j < arc._num_table; ++j) {
//       const auto& table = arc._table[j];
//       CUDA_CHECK(cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._x),
//                             table._num_x * sizeof(double)));
//       CUDA_CHECK(cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._x, table._x,
//                             table._num_x * sizeof(double),
//                             cudaMemcpyHostToDevice));
//       lib_data_gpu._arcs_gpu[i]._table[j]._num_x = table._num_x;

//       CUDA_CHECK(cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._y),
//                             table._num_y * sizeof(double)));
//       CUDA_CHECK(cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._y, table._y,
//                             table._num_y * sizeof(double),
//                             cudaMemcpyHostToDevice));
//       lib_data_gpu._arcs_gpu[i]._table[j]._num_y = table._num_y;

//       CUDA_CHECK(cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._values),
//                             table._num_values * sizeof(double)));
//       CUDA_CHECK(cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._values,
//                             table._values, table._num_values *
//                             sizeof(double), cudaMemcpyHostToDevice));
//       lib_data_gpu._arcs_gpu[i]._table[j]._num_values = table._num_values;
//       lib_data_gpu._arcs_gpu[i]._table[j]._type = table._type;
//     }
//   }
// }

/**
 * @brief print first arc's first table of lib_data_gpu for debug.
 */
void print_lib_data_gpu_first_arc_first_table(LibDataGPU& lib_data_gpu) {
  // gpu->cpu transfer
  LibArcGPU* h_arcs;
  cudaMallocHost(&h_arcs, sizeof(LibArcGPU) * lib_data_gpu._num_arcs);

  cudaMemcpy(h_arcs, lib_data_gpu._arcs_gpu,
             sizeof(LibArcGPU) * lib_data_gpu._num_arcs,
             cudaMemcpyDeviceToHost);

  LibTableGPU* h_table;
  cudaMallocHost(&h_table, sizeof(LibTableGPU) * h_arcs[0]._num_table);
  cudaMemcpy(h_table, h_arcs[0]._table,
             sizeof(LibTableGPU) * h_arcs[0]._num_table,
             cudaMemcpyDeviceToHost);

  LibTableGPU first_table = h_table[0];

  std::cout << "First table values:" << std::endl;
  std::cout << "Num X: " << first_table._num_x << std::endl;
  std::cout << "Num Y: " << first_table._num_y << std::endl;
  std::cout << "Num Values: " << first_table._num_values << std::endl;

  double* h_x;
  double* h_y;
  double* h_values;

  cudaMallocHost(&h_x, sizeof(double) * first_table._num_x);
  cudaMallocHost(&h_y, sizeof(double) * first_table._num_y);
  cudaMallocHost(&h_values, sizeof(double) * first_table._num_values);

  cudaMemcpy(h_x, first_table._x, sizeof(double) * first_table._num_x,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_y, first_table._y, sizeof(double) * first_table._num_y,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_values, first_table._values,
             sizeof(double) * first_table._num_values, cudaMemcpyDeviceToHost);

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

__global__ void kernel_find_value(LibDataGPU* lib_data_gpu, double slew,
                                  double constrain_slew_or_load,
                                  double* d_value) {
  *d_value = find_value(lib_data_gpu->_arcs_gpu[0]._table[0], slew,
                        constrain_slew_or_load);
}

double find_value(LibDataGPU& lib_data_gpu, double slew,
                  double constrain_slew_or_load) {
  // print first arc's first table of lib_data_gpu for debug.
  print_lib_data_gpu_first_arc_first_table(lib_data_gpu);

  // transfer lib_data_gpu_host(host pointer) to lib_data_gpu_device(device
  // pointer) for launching kernel.
  LibDataGPU* lib_data_gpu_host = &lib_data_gpu;

  LibDataGPU* lib_data_gpu_device;
  cudaMalloc((void**)&lib_data_gpu_device, sizeof(LibDataGPU));
  cudaMemcpy(lib_data_gpu_device, lib_data_gpu_host, sizeof(LibDataGPU),
             cudaMemcpyHostToDevice);

  // launch kernel
  double* d_value;
  cudaMalloc((void**)&d_value, sizeof(double));
  kernel_find_value<<<1, 1>>>(lib_data_gpu_device, slew, constrain_slew_or_load,
                              d_value);
  CUDA_CHECK_ERROR();
  cudaDeviceSynchronize();

  double value;
  cudaMemcpy(&value, d_value, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(lib_data_gpu_device);
  cudaFree(d_value);

  return value;
}

}  // namespace ista