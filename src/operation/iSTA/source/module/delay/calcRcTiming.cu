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

#include "ElmoreDelayCalc.hh"
#include "log/Log.hh"

namespace ista {

const int THREAD_PER_BLOCK_NUM = 512;

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
__global__ void kernelUpdateLoad(int* start_array, float* cap_array,
                                 float* ncap_array, float* load_array,
                                 float* nload_array, int* parent_pos_array,
                                 int total_nets_num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < total_nets_num) {
    int current_net = tid;
    int start_node = start_array[current_net];
    int end_node = start_array[current_net + 1];

    for (int i = end_node - 1; i >= start_node; --i) {
      int current_pos = i;
      int parent_pos = parent_pos_array[current_pos];
      // printf("current pos %d current pos cap: %f\n", current_pos,
      //        cap_array[current_pos]);
      float cap = cap_array[current_pos];

      // update the current pos load and parent's load.
      load_array[current_pos] += cap;

      // update the current pos nload and parent's nload.
      nload_array[4 * current_pos + 0] += ncap_array[4 * current_pos + 0];
      nload_array[4 * current_pos + 1] += ncap_array[4 * current_pos + 1];
      nload_array[4 * current_pos + 2] += ncap_array[4 * current_pos + 2];
      nload_array[4 * current_pos + 3] += ncap_array[4 * current_pos + 3];

      if (parent_pos != current_pos) {
        atomicAdd(&load_array[parent_pos], load_array[current_pos]);

        atomicAdd(&nload_array[4 * parent_pos + 0],
                  nload_array[4 * current_pos + 0]);
        atomicAdd(&nload_array[4 * parent_pos + 1],
                  nload_array[4 * current_pos + 1]);
        atomicAdd(&nload_array[4 * parent_pos + 2],
                  nload_array[4 * current_pos + 2]);
        atomicAdd(&nload_array[4 * parent_pos + 3],
                  nload_array[4 * current_pos + 3]);
      }
    }
  }
}

void calcRcTiming(std::vector<RcNet*> all_nets) {
  std::vector<int> start_array{0};
  std::size_t total_nodes_num = 0;
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

  // malloc gpu memory
  cudaMalloc(&gpu_start_array, (total_nodes_num + 1) * sizeof(int));
  cudaMalloc(&gpu_cap_array, total_nodes_num * sizeof(float));
  cudaMalloc(&gpu_ncap_array, 4 * total_nodes_num * sizeof(float));
  cudaMalloc(&gpu_res_array, total_nodes_num * sizeof(float));
  cudaMalloc(&gpu_parent_pos_array, total_nodes_num * sizeof(int));
  // data to be calculated
  cudaMalloc(&gpu_load_array, total_nodes_num * sizeof(float));
  cudaMalloc(&gpu_nload_array, 4 * total_nodes_num * sizeof(float));

  // copy cpu data to gpu memory.
  cudaMemcpy(gpu_start_array, start_array.data(),
             (total_nodes_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_cap_array, cap_array.data(), total_nodes_num * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_ncap_array, ncap_array.data(),
             4 * total_nodes_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_res_array, res_array.data(), total_nodes_num * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_parent_pos_array, parent_pos_array.data(),
             total_nodes_num * sizeof(int), cudaMemcpyHostToDevice);
  // data to be calculated:should initialize to zero
  cudaMemset(gpu_load_array, 0, total_nodes_num * sizeof(float));
  cudaMemset(gpu_nload_array, 0, 4 * total_nodes_num * sizeof(float));

  // launch kernel：kernelUpdateLoad
  auto total_nets_num = all_nets.size();
  int num_blocks =
      (total_nets_num + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;
  // int num_blocks = 2;
  kernelUpdateLoad<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
      gpu_start_array, gpu_cap_array, gpu_ncap_array, gpu_load_array,
      gpu_nload_array, gpu_parent_pos_array, total_nets_num);
  cudaDeviceSynchronize();
  cudaMemcpy(load_array.data(), gpu_load_array, total_nodes_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(nload_array.data(), gpu_nload_array,
             4 * total_nodes_num * sizeof(float), cudaMemcpyDeviceToHost);

  // print_array("load array", load_array);
  // print_array("nload array", nload_array);

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

    rct->set_load_array(net_load_array);
    rct->set_nload_array(net_nload_array);

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
  // cudaFree(gpu_delay_array);
  // cudaFree(gpu_ndelay_array);
  // cudaFree(gpu_ures_array);
  // cudaFree(gpu_ldelay_array);
  // cudaFree(gpu_beta_array);
  // cudaFree(gpu_impulse_array);
}

}  // namespace ista