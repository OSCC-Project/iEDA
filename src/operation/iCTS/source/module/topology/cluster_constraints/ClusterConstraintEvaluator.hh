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
 * @file ClusterConstraintEvaluator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Shared cluster legality and electrical evaluator.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "ClusterConstraintEvaluation.hh"

namespace icts {

struct ClusterConfig;
class Pin;
template <typename T>
class Point;

class ClusterConstraintEvaluator
{
 public:
  ClusterConstraintEvaluator() = default;
  ~ClusterConstraintEvaluator() = default;

  static auto evaluateLoads(const std::vector<Pin*>& loads, const Point<int>& routing_root, const ClusterConfig& config,
                            bool need_exact_cap) -> ConstraintEvaluation;
  static auto evaluatePinnedLoads(const std::vector<Pin*>& loads, std::size_t fanout, int diameter, const Point<int>& routing_root,
                                  const ClusterConfig& config, bool need_exact_cap) -> ConstraintEvaluation;

 private:
  static auto estimatePinCap(const std::vector<Pin*>& loads, const ClusterConfig& config) -> double;
  static auto estimateExactCap(const std::vector<Pin*>& loads, const Point<int>& synthetic_root, const ClusterConfig& config)
      -> ElectricalEstimate;
  static auto queryPinCap(const Pin* pin, const ClusterConfig& config) -> double;
};

}  // namespace icts
