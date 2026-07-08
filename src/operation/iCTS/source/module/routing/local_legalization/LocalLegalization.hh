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
 * @file LocalLegalization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Standalone point-based local legalization.
 */
#pragma once

#include <cstddef>
#include <vector>

#include "Point.hh"
#include "Rect.hh"
#include "Region.hh"

namespace icts {

class LocalLegalization
{
 public:
  using PointType = Point<int>;
  using RectType = Rect<int>;
  using RegionType = Region<int>;

  enum class FailurePolicy
  {
    kFail,
    kKeepOriginal,
  };

  struct Config
  {
    std::size_t candidate_budget = 32;
    int local_search_radius = 6;
    int max_expansion_rounds = 8;
    FailurePolicy failure_policy = FailurePolicy::kFail;
  };

  struct Problem
  {
    std::vector<PointType> movable_points;
    std::vector<PointType> fixed_points;
    RegionType feasible_region;
    RegionType block_region;
  };

  struct Output
  {
    std::vector<PointType> legalized_points;
    long long total_displacement = 0;
    bool success = false;
  };

  LocalLegalization() = delete;
  ~LocalLegalization() = default;

  static auto legalize(const Problem& problem) -> Output;
  static auto legalize(const Problem& problem, const Config& config) -> Output;
  static auto legalize(std::vector<PointType>& movable_points, const std::vector<PointType>& fixed_points = {},
                       const RegionType& feasible_region = RegionType{}, const RegionType& block_region = RegionType{}) -> Output;
  static auto legalize(std::vector<PointType>& movable_points, const std::vector<PointType>& fixed_points,
                       const RegionType& feasible_region, const RegionType& block_region, const Config& config) -> Output;

 private:
  struct CandidateSite
  {
    PointType point = PointType(-1, -1);
  };

  static auto buildLegalRegion(const Problem& problem) -> RegionType;

  static auto generateCandidates(const PointType& origin, const RegionType& legal_region, const std::vector<PointType>& fixed_points,
                                 std::size_t candidate_budget, int local_search_radius) -> std::vector<CandidateSite>;
  static auto enumerateProjectedNeighbors(const PointType& seed, const RegionType& legal_region, const std::vector<PointType>& fixed_points,
                                          int max_radius, std::size_t candidate_budget) -> std::vector<CandidateSite>;
  static auto enumerateBoundaryBreakpoints(const PointType& origin, const RegionType& legal_region,
                                           const std::vector<PointType>& fixed_points, std::size_t candidate_budget)
      -> std::vector<CandidateSite>;
  static auto solveAssignment(const std::vector<PointType>& movable_points, const std::vector<std::vector<CandidateSite>>& candidate_sets)
      -> std::vector<PointType>;
  static auto computeTotalDisplacement(const std::vector<PointType>& original_points, const std::vector<PointType>& legalized_points)
      -> long long;
  static auto appendCandidate(std::vector<CandidateSite>& candidates, const PointType& point, const RegionType& legal_region,
                              const std::vector<PointType>& fixed_points, std::size_t candidate_budget) -> void;
};

}  // namespace icts
