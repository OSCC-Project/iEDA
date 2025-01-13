/**
 * @file LibArc.cu
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement lookup table.
 * @version 0.1
 * @date 2025-01-09
 */
#include <cassert>

#include "LibArc.cuh"
#include "Type.hh"
#include "log/Log.hh"
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
                                     double x) {
  assert(!IsDoubleEqual(x1, x2));

  if (x >= std::numeric_limits<double>::max() ||
      x <= std::numeric_limits<double>::lowest()) {
    return x;
  }

  double slope = (y2 - y1) / (x2 - x1);
  double ret_val;

  if (x < x1) {
    ret_val = y1 - (x1 - x) * slope;  // Extrapolation.
  } else if (x > x2) {
    ret_val = y2 + (x - x2) * slope;  // Extrapolation.
  } else if (IsDoubleEqual(x, x1)) {
    ret_val = y1;  // Boundary case.
  } else if (IsDoubleEqual(x, x2)) {
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
 * @brief find value according to slew and constrain_slew_or_load.
 * @param slew The slew value.
 * @param constrain_slew_or_load The constrain_slew_or_load value.
 * @return The value.
 */
__device__ double LibTableGPU::find_value(double slew,
                                          double constrain_slew_or_load) {
  // ??? not sure (_type == UINT_MAX) can work as (!table_template)
  if (_type == UINT_MAX) {
    return _values[0];
  }

  double val1;
  double val2;
  if (_type == 0 || _type == 2) {
    val1 = slew;
    val2 = constrain_slew_or_load;
  } else if (_type == 1 || _type == 3) {
    val1 = constrain_slew_or_load;
    val2 = slew;
  } else {
    // ??? not sure (_type == UINT_MAX) can work as (!table_template)
    LOG_FATAL << "lut table: invalid delay lut template variable";
  }

  if (_num_y == 0) {
    // only one axis(x axis.)
    auto num_val1 = check_val(0, val1);
    auto [x1, x2, val1_index] = get_axis_region(0, num_val1, val1);
    unsigned int x1_table_val = get_table_value(val1_index);
    unsigned int x2_table_val = get_table_value(val1_index + 1);

    auto result = linear_interpolate(x1, x2, x1_table_val, x2_table_val, val1);
    return result;
  } else {
    // two axis(x and y axis.)
    auto num_val1 = check_val(0, val1);
    auto num_val2 = check_val(1, val2);

    auto [x1, x2, val1_index] = get_axis_region(0, num_val1, val1);
    auto [y1, y2, val2_index] = get_axis_region(1, num_val2, val2);

    // now do the table lookup
    unsigned int index = num_val2 * val1_index + val2_index;
    const auto q11 = get_table_value(index);

    index = num_val2 * (val1_index + 1) + val2_index;
    const auto q21 = get_table_value(index);

    index = num_val2 * val1_index + (val2_index + 1);
    const auto q12 = get_table_value(index);

    index = num_val2 * (val1_index + 1) + (val2_index + 1);
    const auto q22 = get_table_value(index);

    auto result =
        bilinear_interpolation(q11, q12, q21, q22, x1, x2, y1, y2, val1, val2);

    // LOG_ERROR_IF_EVERY_N(result < 0.0, 100) << "table " << get_file_name() <<
    // " " << get_line_no() << " "
    //                                         << "delay value less zero.";
    return result;
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

  lib_data_gpu._num_arcs = lib_arcs_cpu.size();
  // ???  not sure need _arcs as member of LibDataGPU
  lib_data_gpu._arcs = lib_arcs_cpu;
  // copy cpu data(lib_arcs_cpu) to gpu data(lib_data_gpu._arcs_gpu)
  cudaMalloc(&(lib_data_gpu._arcs_gpu),
             lib_data_gpu._num_arcs * sizeof(LibArcGPU));

  cudaMemcpy(lib_data_gpu._arcs_gpu, lib_data_gpu._arcs.data(),
             lib_data_gpu._num_arcs * sizeof(LibArcGPU),
             cudaMemcpyHostToDevice);

  // copy table data to gpu
  for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
    const auto& arc = lib_data_gpu._arcs[i];
    cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table),
               arc._num_table * sizeof(LibTableGPU));
    cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table, arc._table,
               arc._num_table * sizeof(LibTableGPU), cudaMemcpyHostToDevice);
    lib_data_gpu._arcs_gpu[i]._num_table = arc._num_table;
  }

  // copy table data to gpu,copy x,y and values to gpu
  for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
    const auto& arc = lib_data_gpu._arcs[i];
    for (unsigned j = 0; j < arc._num_table; ++j) {
      const auto& table = arc._table[j];
      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._x),
                 table._num_x * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._x, table._x,
                 table._num_x * sizeof(double), cudaMemcpyHostToDevice);
      lib_data_gpu._arcs_gpu[i]._table[j]._num_x = table._num_x;

      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._y),
                 table._num_y * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._y, table._y,
                 table._num_y * sizeof(double), cudaMemcpyHostToDevice);
      lib_data_gpu._arcs_gpu[i]._table[j]._num_y = table._num_y;

      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._values),
                 table._num_values * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._values, table._values,
                 table._num_values * sizeof(double), cudaMemcpyHostToDevice);
      lib_data_gpu._arcs_gpu[i]._table[j]._num_values = table._num_values;
      lib_data_gpu._arcs_gpu[i]._table[j]._type = table._type;
    }
  }
}

}  // namespace ista