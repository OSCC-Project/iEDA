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
 * @file BottomUpMergeInfeasibility.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Stage B.3 · Infeasibility fallback (transformed-rectangle merge
 *        region, min-skew section, detour edge) for bound-skew trees.
 *        Used when the joining region has no point that satisfies the skew
 *        bound during bottom-up merging.
 */

#pragma once

namespace icts::bst {
class Area;
}  // namespace icts::bst

namespace icts::bst::detail {

class BoundSkewTreeImpl;

class BottomUpMergeInfeasibility
{
 public:
  explicit BottomUpMergeInfeasibility(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~BottomUpMergeInfeasibility() = default;
  BottomUpMergeInfeasibility(const BottomUpMergeInfeasibility&) = delete;
  auto operator=(const BottomUpMergeInfeasibility&) -> BottomUpMergeInfeasibility& = delete;

  auto constructInfeasibleMergeRegion(Area* parent) const -> void;
  auto constructTransformedRectMergeRegion(Area* current_area) const -> void;

 private:
  auto calcMinSkewSection(Area* current_area) const -> void;
  auto calcDetourEdgeLength(Area* current_area) const -> void;
  auto refineMergeRegionDelay(Area* current_area) const -> void;

  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
