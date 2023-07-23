// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file BalanceClustering.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <vector>

#include "Node.hh"

namespace icts {
/**
 * @brief BalanceClustering class
 *       clustering sinks by max distance and max fanout
 *       using k-means algorithm construct a initial clustering
 *       then iteratively adjust the clustering by min cost flow to balance the capacitance variance
 *
 */
class BalanceClustering
{
 public:
  BalanceClustering() = delete;
  ~BalanceClustering() = default;
  static std::vector<std::vector<Node*>> kMeans(const std::vector<Node*>& sinks, const size_t& k, const int& seed = 0,
                                                const size_t& max_iter = 500);

  static std::vector<std::vector<Node*>> iterClustering(const std::vector<Node*>& sinks, const double& max_dist, const size_t& max_fanout,
                                                        const size_t& iters = 100, const size_t& no_change_stop = 10,
                                                        const double& limit_ratio = 0.8);

  static std::vector<Point> genCentroidBuffers(const std::vector<std::vector<Node*>>& clusters);

  static double calcSAE(const std::vector<std::vector<Node*>>& clusters, const std::vector<Point>& buffers);

  static double calcCapVariance(const std::vector<std::vector<Node*>>& clusters, const std::vector<Point>& buffers);
};
}  // namespace icts