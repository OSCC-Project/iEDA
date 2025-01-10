/**
 * @file LibArc.cu
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement lookup table.
 * @version 0.1
 * @date 2025-01-09
 */
#include "LibArc.cuh"
namespace ista {

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
    // set gpu arc's table num_table
  }

  // copy table data to gpu
  // copy x,y and values to gpu
  for (unsigned i = 0; i < lib_data_gpu._num_arcs; ++i) {
    const auto& arc = lib_data_gpu._arcs[i];
    for (unsigned j = 0; j < arc._num_table; ++j) {
      const auto& table = arc._table[j];
      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._x),
                 table._num_x * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._x, table._x,
                 table._num_x * sizeof(double), cudaMemcpyHostToDevice);

      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._y),
                 table._num_y * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._y, table._y,
                 table._num_y * sizeof(double), cudaMemcpyHostToDevice);

      cudaMalloc(&(lib_data_gpu._arcs_gpu[i]._table[j]._values),
                 table._num_values * sizeof(double));
      cudaMemcpy(lib_data_gpu._arcs_gpu[i]._table[j]._values, table._values,
                 table._num_values * sizeof(double), cudaMemcpyHostToDevice);
    }
  }
}

}  // namespace ista