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
 * @file AnnealOpt.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <limits>
#include <ranges>

#include "Inst.hh"

namespace icts {
/**
 * @brief AnnealOpt for rebalance clustering
 *       input: clusters
 *       constraint: net_num, max_fanout, max_cap, max_net_dist,
 *                   p(coefficient of net length), q(coefficient of net skew)
 *       output: new clusters which satisfy the constraints
 *
 */
class AnnealOpt
{
 public:
  AnnealOpt(const std::vector<std::vector<Inst*>>& clusters) : _clusters(clusters)
  {
    std::ranges::for_each(clusters, [&](const std::vector<Inst*>& cluster) {
      std::ranges::for_each(cluster, [&](Inst* inst) { _flatten_insts.push_back(inst); });
    });
  };
  ~AnnealOpt() = default;
  // init parameters
  void initParameter(const size_t& net_num, const int& max_fanout, const double& max_cap, const int& max_net_dist, const double& p,
                     const double& q, const double& r);

  // run
  std::vector<std::vector<Inst*>> run();

 private:
  /**
   * @brief Database and parameters
   *
   */
  std::vector<std::vector<Inst*>> _clusters;
  std::vector<Inst*> _flatten_insts;

  int _max_fanout = 0;
  double _max_cap = 0;
  int _max_net_dist = 0;
  double _p = 0;
  double _q = 0;
  double _r = 0;

  size_t _net_num = 0;
  size_t _inst_num = 0;
  int _min_x = std::numeric_limits<int>::max();
  int _min_y = std::numeric_limits<int>::max();
  int _max_x = std::numeric_limits<int>::min();
  int _max_y = std::numeric_limits<int>::min();
  const double _lambda = 1e8;
};
}  // namespace icts