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
 * @file ClusterConstraintEvaluation.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Shared cluster legality and electrical evaluation data types.
 */

#pragma once

#include <cmath>
#include <cstddef>

#include "spatial/Point.hh"

namespace icts {

struct ElectricalEstimate
{
  bool exact = false;
  bool route_success = false;
  Point<int> synthetic_root = Point<int>(0, 0);
  Point<int> legalized_root = Point<int>(0, 0);
  Point<int> routed_root = Point<int>(0, 0);
  double pin_cap = 0.0;
  double wire_cap = 0.0;
  double total_cap = 0.0;
  double wirelength = 0.0;
};

struct ClusterConstraintMetrics
{
  std::size_t fanout = 0;
  int diameter = 0;
  double cap_lower_bound = 0.0;
  double total_cap = 0.0;
  double wirelength = 0.0;
  ElectricalEstimate electrical;
};

enum class ConstraintViolation
{
  kNone,
  kEmptyCluster,
  kFanout,
  kDiameter,
  kCapacitance,
  kRoutingFailed,
};

struct ConstraintEvaluation
{
  bool legal = false;
  ConstraintViolation violation = ConstraintViolation::kEmptyCluster;
  ClusterConstraintMetrics metrics;
};

inline auto IsFiniteCapLimit(double max_cap) -> bool
{
  return max_cap > 0.0 && std::isfinite(max_cap);
}

}  // namespace icts
