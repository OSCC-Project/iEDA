// ***************************************************************************************
// MIT License
//
// Copyright (c) 2018-2021 Tsung-Wei Huang and Martin D. F. Wong
//
// The University of Utah, UT, USA
//
// The University of Illinois at Urbana-Champaign, IL, USA
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ***************************************************************************************
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
 * @file calcRcTiming.cu
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement traditional elmore calc algorithm.
 * @version 0.1
 * @date 2024-12-13
 */

#include <cuda_runtime.h>

#include "delay/ElmoreDelayCalc.hh"
#include "gpu/cuda_common.cuh"
#include "usage/usage.hh"

namespace ista {

// for gpu threads constrain, calc net max num per epoch
__constant__ unsigned c_max_net_per_epoch;

/**
 * @brief gpu speed up update the load of the rc node.
 *
 * @param start_array
 * @param cap_array
 * @param ncap_array
 * @param load_array
 * @param nload_array
 * @param parent_pos_arrays
 * @param total_nets_num
 * @return __global__
 */
__global__ void kernel_update_load(int* start_array, float* cap_array,
                                   float* ncap_array, float* load_array,
                                   float* nload_array, int* parent_pos_array,
                                   int total_nets_num, int current_epoch) {
  int net_tid = blockIdx.x * blockDim.x + threadIdx.x +
                current_epoch * c_max_net_per_epoch;
  int cond_tid = threadIdx.y;

  if (net_tid < total_nets_num && cond_tid < 4) {
    int current_net = net_tid;
    int start_node = start_array[current_net];
    int end_node = start_array[current_net + 1];

    for (int i = end_node - 1; i >= start_node; --i) {
      int current_pos = i;
      int parent_pos = parent_pos_array[current_pos];
      // printf("current pos %d current pos cap: %f\n", current_pos,
      //        cap_array[current_pos]);

      if (cond_tid == 0) {
        float cap = cap_array[current_pos];
        // update the current pos load and parent's load.
        load_array[current_pos] += cap;
      }

      // update the current pos nload and parent's nload.
      nload_array[4 * current_pos + cond_tid] +=
          ncap_array[4 * current_pos + cond_tid];

      if (parent_pos != current_pos) {
        if (cond_tid == 0) {
          atomicAdd(&load_array[parent_pos], load_array[current_pos]);
        }

        atomicAdd(&nload_array[4 * parent_pos + cond_tid],
                  nload_array[4 * current_pos + cond_tid]);
      }
    }
  }
}

/**
 * @brief gpu speed up update the delay of the rc node.
 *
 * @param start_array
 * @param res_array
 * @param load_array
 * @param nload_array
 * @param delay_array
 * @param ndelay_array
 * @param parent_pos_arrays
 * @param total_nets_num
 * @return __global__
 */
__global__ void kernel_update_delay(int* start_array, float* res_array,
                                    float* load_array, float* nload_array,
                                    float* delay_array, float* ndelay_array,
                                    float* ures_array, int* parent_pos_array,
                                    int total_nets_num, int current_epoch) {
  int net_tid = blockIdx.x * blockDim.x + threadIdx.x +
                current_epoch * c_max_net_per_epoch;
  int cond_tid = threadIdx.y;

  if (net_tid < total_nets_num && cond_tid < 4) {
    int current_net = net_tid;
    int start_node = start_array[current_net];
    int end_node = start_array[current_net + 1];

    // root point delay/ndelay/ures is 0.0.
    for (int i = start_node + 1; i < end_node; ++i) {
      int current_pos = i;
      int parent_pos = parent_pos_array[current_pos];
      float res = res_array[current_pos];

      if (cond_tid == 0) {
        // printf("parent pos %d parent delay cap: %f\n", parent_pos,
        //        delay_array[parent_pos]);
        float parent_delay = delay_array[parent_pos];
        // update delay_array
        float load = load_array[current_pos];
        float delay = parent_delay + res * load;
        atomicAdd(&delay_array[current_pos], delay);
      }

      // update ndelay_array
      float parent_ndelay = ndelay_array[4 * parent_pos + cond_tid];
      float nload = nload_array[4 * current_pos + cond_tid];
      float ndelay = parent_ndelay + res * nload;
      atomicAdd(&ndelay_array[4 * current_pos + cond_tid], ndelay);

      // update ures_array
      float parent_ures = ures_array[4 * parent_pos + cond_tid];
      float ures = parent_ures + res;
      atomicAdd(&ures_array[4 * current_pos + cond_tid], ures);
    }
  }
}

/**
 * @brief gpu speed up update the ldelay of the rc node.
 *
 * @param start_array
 * @param ncap_array
 * @param ndelay_array
 * @param ldelay_array
 * @param parent_pos_array
 * @param total_nets_num
 * @return __global__
 */
__global__ void kernel_update_ldelay(int* start_array, float* ncap_array,
                                     float* ndelay_array, float* ldelay_array,
                                     int* parent_pos_array, int total_nets_num,
                                     int current_epoch) {
  int net_tid = blockIdx.x * blockDim.x + threadIdx.x +
                current_epoch * c_max_net_per_epoch;
  int cond_tid = threadIdx.y;

  if (net_tid < total_nets_num && cond_tid < 4) {
    int current_net = net_tid;
    int start_node = start_array[current_net];
    int end_node = start_array[current_net + 1];

    for (int i = end_node - 1; i >= start_node; --i) {
      int current_pos = i;
      int parent_pos = parent_pos_array[current_pos];

      // update the current pos nload and parent's nload.
      ldelay_array[4 * current_pos + cond_tid] +=
          ncap_array[4 * current_pos + cond_tid] *
          ndelay_array[4 * current_pos + cond_tid];

      if (parent_pos != current_pos) {
        atomicAdd(&ldelay_array[4 * parent_pos + cond_tid],
                  ldelay_array[4 * current_pos + cond_tid]);
      }
    }
  }
}

/**
 * @brief gpu speed up update the response of the rc node.
 *
 * @param res_array
 * @param ldelay_array
 * @param ndelay_array
 * @param beta_array
 * @param impulse_array
 *@param parent_pos_array
 * @param total_nets_num
 * @return __global__
 */
__global__ void kernel_update_response(int* start_array, float* res_array,
                                       float* ldelay_array, float* ndelay_array,
                                       float* beta_array, float* impulse_array,
                                       int* parent_pos_array,
                                       int total_nets_num, int current_epoch) {
  int net_tid = blockIdx.x * blockDim.x + threadIdx.x +
                current_epoch * c_max_net_per_epoch;
  int cond_tid = threadIdx.y;

  if (net_tid < total_nets_num && cond_tid < 4) {
    int current_net = net_tid;
    int start_node = start_array[current_net];
    int end_node = start_array[current_net + 1];

    // root point beta/impulse is 0.0.
    for (int i = start_node + 1; i < end_node; ++i) {
      int current_pos = i;
      int parent_pos = parent_pos_array[current_pos];
      float res = res_array[current_pos];

      // update beta_array
      float parent_beta = beta_array[4 * parent_pos + cond_tid];
      float ldelay = ldelay_array[4 * current_pos + cond_tid];
      float beta = parent_beta + res * ldelay;
      atomicAdd(&beta_array[4 * current_pos + cond_tid], beta);

      // update impulse_array
      float current_ndelay = ndelay_array[4 * current_pos + cond_tid];
      float impulse = 2 * beta - std::pow(current_ndelay, 2);
      atomicAdd(&impulse_array[4 * current_pos + cond_tid], impulse);
    }
  }
}

/**
 * @brief calculate the rc timing of all nets in parallel
 *
 * @param  std::vector<RcNet*> all_nets
 * @return void
 */
void calc_rc_timing(std::vector<RcNet*> all_nets) {
  ieda::Stats stats;
  LOG_INFO << "calc rc timing start";

  std::vector<int> start_array{0};
  unsigned long total_nodes_num = 0;
  // start_array: last element is the end position of the last tree.
  for (const auto& net : all_nets) {
    auto rct = net->rct();
    int node_num = rct->numNodes();
    total_nodes_num += node_num;
    start_array.emplace_back(total_nodes_num);
  }

  std::vector<float> cap_array;
  std::vector<float> ncap_array;
  std::vector<float> load_array(total_nodes_num, 0);
  std::vector<float> nload_array(total_nodes_num * 4, 0);
  std::vector<int> parent_pos_array;
  std::vector<int> children_pos_array(total_nodes_num * 2, 0);
  std::vector<float> res_array;
  std::vector<float> delay_array(total_nodes_num, 0);
  std::vector<float> ndelay_array(total_nodes_num * 4, 0);
  std::vector<float> ures_array(total_nodes_num * 4, 0);
  std::vector<float> ldelay_array(total_nodes_num * 4, 0);
  std::vector<float> beta_array(total_nodes_num * 4, 0);
  std::vector<float> impulse_array(total_nodes_num * 4, 0);

  // for debugging(print_array("cap_array", cap_array);)
  // auto print_array = [](const std::string& array_name,
  //                       const std::vector<float>& array) {
  //   std::cout << array_name << " contents: ";
  //   for (const auto& element : array) {
  //     std::cout << element << " ";
  //     std::cout << std::endl;
  //   };

  // for debugging(print_array("parent_pos_array", parent_pos_array);)
  //   auto print_int_array = [](const std::string& array_name,
  //                             const std::vector<int>& array) {
  //     std::cout << array_name << " contents: ";
  //     for (const auto& element : array) {
  //       std::cout << element << " ";
  //     }
  //     std::cout << std::endl << std::endl;
  //   };

  for (std::size_t index = 0; index < all_nets.size(); ++index) {
    auto net = all_nets[index];
    auto rct = net->rct();
    cap_array.insert(cap_array.end(), rct->get_cap_array().begin(),
                     rct->get_cap_array().end());
    // std::cout << "net_name: " << net->name()
    //           << "; rct cap_array size:" << rct->get_cap_array().size()
    //           << std::endl;
    // print_array("rct cap_array", rct->get_cap_array());

    // std::cout << "cap_array size:" << cap_array.size() << std::endl;
    // print_array("cap_array", cap_array);
    ncap_array.insert(ncap_array.end(), rct->get_ncap_array().begin(),
                      rct->get_ncap_array().end());

    // parent_pos_array requires adding offset.
    auto parent_pos = rct->get_parent_pos_array();
    for (auto& pos : parent_pos) {
      pos += start_array[index];
    }
    parent_pos_array.insert(parent_pos_array.end(), parent_pos.begin(),
                            parent_pos.end());
    res_array.insert(res_array.end(), rct->get_res_array().begin(),
                     rct->get_res_array().end());
  }

  LOG_INFO << "start alloc gpu memory and copy cpu data to gpu memory";

  // initGpuMemory
  int* gpu_start_array =
      nullptr;  // the offsets of each net in arrays of nodes on gpu.
  float* gpu_cap_array = nullptr;       // cap located on gpu.
  float* gpu_ncap_array = nullptr;      // ncap located on gpu.
  float* gpu_res_array = nullptr;       // res located on gpu.
  int* gpu_parent_pos_array = nullptr;  // parent pos located on gpu.
  // need calculate
  float* gpu_load_array = nullptr;
  float* gpu_nload_array = nullptr;
  float* gpu_delay_array = nullptr;
  float* gpu_ndelay_array = nullptr;
  float* gpu_ures_array = nullptr;
  float* gpu_ldelay_array = nullptr;
  float* gpu_beta_array = nullptr;
  float* gpu_impulse_array = nullptr;

  cudaStream_t stream1 = nullptr;
  cudaStreamCreate(&stream1);
  // malloc gpu memory
  CUDA_CHECK(cudaMallocAsync(&gpu_start_array,
                             (total_nodes_num + 1) * sizeof(int), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_cap_array, total_nodes_num * sizeof(float),
                             stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_ncap_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_res_array, total_nodes_num * sizeof(float),
                             stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_parent_pos_array,
                             total_nodes_num * sizeof(int), stream1));
  // data to be calculated
  CUDA_CHECK(cudaMallocAsync(&gpu_load_array, total_nodes_num * sizeof(float),
                             stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_nload_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_delay_array, total_nodes_num * sizeof(float),
                             stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_ndelay_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_ures_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_ldelay_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_beta_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&gpu_impulse_array,
                             4 * total_nodes_num * sizeof(float), stream1));
  // wait for all cudaMallocAsync to complete
  CUDA_CHECK(cudaStreamSynchronize(stream1));

  CUDA_CHECK_ERROR();

  // copy cpu data to gpu memory.
  CUDA_CHECK(cudaMemcpyAsync(gpu_start_array, start_array.data(),
                             (total_nodes_num + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(gpu_cap_array, cap_array.data(),
                             total_nodes_num * sizeof(float),
                             cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(gpu_ncap_array, ncap_array.data(),
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(gpu_res_array, res_array.data(),
                             total_nodes_num * sizeof(float),
                             cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(gpu_parent_pos_array, parent_pos_array.data(),
                             total_nodes_num * sizeof(int),
                             cudaMemcpyHostToDevice, stream1));
  // data to be calculated:should initialize to zero
  cudaMemsetAsync(gpu_load_array, 0, total_nodes_num * sizeof(float), stream1);
  cudaMemsetAsync(gpu_nload_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  cudaMemsetAsync(gpu_delay_array, 0, total_nodes_num * sizeof(float), stream1);
  cudaMemsetAsync(gpu_ndelay_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  cudaMemsetAsync(gpu_ures_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  cudaMemsetAsync(gpu_ldelay_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  cudaMemsetAsync(gpu_beta_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  cudaMemsetAsync(gpu_impulse_array, 0, 4 * total_nodes_num * sizeof(float),
                  stream1);
  CUDA_CHECK(cudaStreamSynchronize(stream1));

  LOG_INFO << "finish alloc gpu memory and copy cpu data to gpu memory";

  unsigned max_net_per_epoch = 5120;
  // unsigned max_net_per_epoch = 2560;
  cudaMemcpyToSymbol(c_max_net_per_epoch, &max_net_per_epoch, sizeof(unsigned), 0,
                     cudaMemcpyHostToDevice);  // Copy to constant memory

  // launch kernel：kernelUpdateLoad
  auto total_nets_num = all_nets.size();
  dim3 block_size(256, 4);
  dim3 num_blocks((max_net_per_epoch + block_size.x - 1) / block_size.x, 1);

  unsigned num_gpu_threads =
      num_blocks.x * num_blocks.y * block_size.x * block_size.y;

  LOG_INFO << "start gpu kernel to calc elmore delay threads num "
           << num_gpu_threads;

  printf(
      "use grid dims (%d, %d) with block dims (%d, %d), total threads: %d, "
      "start run gpu kernel to speedup elmore delay.\n",
      num_blocks.x, num_blocks.y, block_size.x, block_size.y, num_gpu_threads);

  int num_epoch = (total_nets_num + max_net_per_epoch - 1) / max_net_per_epoch;
  for (int i = 0; i < num_epoch; ++i) {
    LOG_INFO << "epoch " << i + 1 << " for total epoch " << num_epoch;

    kernel_update_load<<<num_blocks, block_size>>>(
        gpu_start_array, gpu_cap_array, gpu_ncap_array, gpu_load_array,
        gpu_nload_array, gpu_parent_pos_array, total_nets_num, i);
    kernel_update_delay<<<num_blocks, block_size>>>(
        gpu_start_array, gpu_res_array, gpu_load_array, gpu_nload_array,
        gpu_delay_array, gpu_ndelay_array, gpu_ures_array, gpu_parent_pos_array,
        total_nets_num, i);
    kernel_update_ldelay<<<num_blocks, block_size>>>(
        gpu_start_array, gpu_ncap_array, gpu_ndelay_array, gpu_ldelay_array,
        gpu_parent_pos_array, total_nets_num, i);
    kernel_update_response<<<num_blocks, block_size>>>(
        gpu_start_array, gpu_res_array, gpu_ldelay_array, gpu_ndelay_array,
        gpu_beta_array, gpu_impulse_array, gpu_parent_pos_array,
        total_nets_num, i);
  }

  // get launch kernel error status
  CUDA_CHECK_ERROR();

  cudaDeviceSynchronize();

  LOG_INFO << "finish gpu kernel to calc elmore delay threads num "
           << num_gpu_threads;

  LOG_INFO << "start copy gpu data to cpu memory";
  // LOG_INFO << "calculate rc timing end";
  // // LOG_INFO << "calculate rc timing net num: " << all_nets.size();
  // double memory_delta = stats.memoryDelta();
  // LOG_INFO << "calculate rc timing memory usage " << memory_delta << "MB";
  // double time_delta = stats.elapsedRunTime();
  // LOG_INFO << "calculate rc timing time elapsed " << time_delta << "s";
  // copy gpu data to cpu memory.
  CUDA_CHECK(cudaMemcpyAsync(load_array.data(), gpu_load_array,
                             total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(nload_array.data(), gpu_nload_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(delay_array.data(), gpu_delay_array,
                             total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(ndelay_array.data(), gpu_ndelay_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(ures_array.data(), gpu_ures_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(ldelay_array.data(), gpu_ldelay_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(beta_array.data(), gpu_beta_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(impulse_array.data(), gpu_impulse_array,
                             4 * total_nodes_num * sizeof(float),
                             cudaMemcpyDeviceToHost, stream1));
  cudaStreamSynchronize(stream1);
  cudaStreamDestroy(stream1);
  // print_array("load array", load_array);
  // print_array("nload array", nload_array);

  CUDA_CHECK_ERROR();

  std::size_t offset = 0;
  for (const auto& net : all_nets) {
    auto rct = net->rct();
    int node_num = rct->numNodes();
    std::vector<float> net_load_array(load_array.begin() + offset,
                                      load_array.begin() + offset + node_num);
    std::vector<float> net_nload_array(
        nload_array.begin() + offset * 4,
        nload_array.begin() + (offset + node_num) * 4);
    // printf("net name:%s", net->name().c_str());
    // print_array("net_load_array", net_load_array);
    // print_array("net_nload_array", net_nload_array);
    std::vector<float> net_delay_array(delay_array.begin() + offset,
                                       delay_array.begin() + offset + node_num);
    std::vector<float> net_ndelay_array(
        ndelay_array.begin() + offset * 4,
        ndelay_array.begin() + (offset + node_num) * 4);
    std::vector<float> net_ures_array(
        ures_array.begin() + offset * 4,
        ures_array.begin() + (offset + node_num) * 4);
    std::vector<float> net_ldelay_array(
        ldelay_array.begin() + offset * 4,
        ldelay_array.begin() + (offset + node_num) * 4);
    std::vector<float> net_beta_array(
        beta_array.begin() + offset * 4,
        beta_array.begin() + (offset + node_num) * 4);
    std::vector<float> net_impulse_array(
        impulse_array.begin() + offset * 4,
        impulse_array.begin() + (offset + node_num) * 4);

    rct->set_load_array(net_load_array);
    rct->set_nload_array(net_nload_array);
    rct->set_delay_array(net_delay_array);
    rct->set_ndelay_array(net_ndelay_array);
    rct->set_ures_array(net_ures_array);
    rct->set_ldelay_array(net_ldelay_array);
    rct->set_beta_array(net_beta_array);
    rct->set_impulse_array(net_impulse_array);

    offset += node_num;
  }

  // set node according to the array.
  for (const auto& net : all_nets) {
    auto rct = net->rct();
    auto& rct_load_array = rct->get_load_array();
    auto& rct_nload_array = rct->get_nload_array();
    // printf("net name:%s", net->name().c_str());
    // print_array("rct_load_array", rct_load_array);
    // print_array("rct_nload_array", rct_nload_array);
    auto& rct_delay_array = rct->get_delay_array();
    auto& rct_ndelay_array = rct->get_ndelay_array();
    auto& rct_ures_array = rct->get_ures_array();
    auto& rct_ldelay_array = rct->get_ldelay_array();
    auto& rct_beta_array = rct->get_beta_array();
    auto& rct_impulse_array = rct->get_impulse_array();

    for (auto& [node_name, node] : rct->get_nodes()) {
      // set the node load.
      node.set_load(rct_load_array[node.get_flatten_pos()]);
      // std::cout << "node load: " << node.nodeLoad() << std::endl;
      // set the node nload.
      auto& node_nload = node.get_nload();
      node_nload[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_nload_array[4 * node.get_flatten_pos() + 0];
      node_nload[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_nload_array[4 * node.get_flatten_pos() + 1];
      node_nload[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_nload_array[4 * node.get_flatten_pos() + 2];
      node_nload[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_nload_array[4 * node.get_flatten_pos() + 3];
      // std::cout
      //     << "node nload: "
      //     << node_nload[ModeTransPair(AnalysisMode::kMax, TransType::kRise)]
      //     << ","
      //     << node_nload[ModeTransPair(AnalysisMode::kMax, TransType::kFall)]
      //     << ","
      //     << node_nload[ModeTransPair(AnalysisMode::kMin, TransType::kRise)]
      //     << ","
      //     << node_nload[ModeTransPair(AnalysisMode::kMin, TransType::kRise)]
      //     << "," << std::endl;

      // set the node delay.
      node.set_delay(rct_delay_array[node.get_flatten_pos()]);
      // set the node ndelay.
      auto& node_ndelay = node.get_ndelay();
      node_ndelay[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_ndelay_array[4 * node.get_flatten_pos() + 0];
      node_ndelay[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_ndelay_array[4 * node.get_flatten_pos() + 1];
      node_ndelay[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_ndelay_array[4 * node.get_flatten_pos() + 2];
      node_ndelay[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_ndelay_array[4 * node.get_flatten_pos() + 3];

      // set the node ures.
      auto& node_ures = node.get_ures();
      node_ures[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_ures_array[4 * node.get_flatten_pos() + 0];
      node_ures[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_ures_array[4 * node.get_flatten_pos() + 1];
      node_ures[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_ures_array[4 * node.get_flatten_pos() + 2];
      node_ures[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_ures_array[4 * node.get_flatten_pos() + 3];

      // set the node ldelay.
      auto& node_ldelay = node.get_ldelay();
      node_ldelay[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_ldelay_array[4 * node.get_flatten_pos() + 0];
      node_ldelay[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_ldelay_array[4 * node.get_flatten_pos() + 1];
      node_ldelay[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_ldelay_array[4 * node.get_flatten_pos() + 2];
      node_ldelay[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_ldelay_array[4 * node.get_flatten_pos() + 3];

      // set the node beta.
      auto& node_beta = node.get_beta();
      node_beta[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_beta_array[4 * node.get_flatten_pos() + 0];
      node_beta[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_beta_array[4 * node.get_flatten_pos() + 1];
      node_beta[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_beta_array[4 * node.get_flatten_pos() + 2];
      node_beta[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_beta_array[4 * node.get_flatten_pos() + 3];

      // set the node impulse.
      auto& node_impulse = node.get_impulse();
      node_impulse[ModeTransPair(AnalysisMode::kMax, TransType::kRise)] =
          rct_impulse_array[4 * node.get_flatten_pos() + 0];
      node_impulse[ModeTransPair(AnalysisMode::kMax, TransType::kFall)] =
          rct_impulse_array[4 * node.get_flatten_pos() + 1];
      node_impulse[ModeTransPair(AnalysisMode::kMin, TransType::kRise)] =
          rct_impulse_array[4 * node.get_flatten_pos() + 2];
      node_impulse[ModeTransPair(AnalysisMode::kMin, TransType::kFall)] =
          rct_impulse_array[4 * node.get_flatten_pos() + 3];
    }
  }

  // free gpu memory.
  cudaFree(gpu_start_array);
  cudaFree(gpu_cap_array);
  cudaFree(gpu_ncap_array);
  cudaFree(gpu_res_array);
  cudaFree(gpu_parent_pos_array);
  cudaFree(gpu_load_array);
  cudaFree(gpu_nload_array);
  cudaFree(gpu_delay_array);
  cudaFree(gpu_ndelay_array);
  cudaFree(gpu_ures_array);
  cudaFree(gpu_ldelay_array);
  cudaFree(gpu_beta_array);
  cudaFree(gpu_impulse_array);

  LOG_INFO << "finish copy gpu data to cpu memory and free gpu memory";

  LOG_INFO << "calc rc timing end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "calc rc timing " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "calc rc timing " << time_delta << "s";
}

}  // namespace ista