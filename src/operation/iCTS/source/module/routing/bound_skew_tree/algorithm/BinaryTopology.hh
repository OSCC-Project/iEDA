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
 * @file BinaryTopology.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Stage A · Binary topology construction for bound-skew trees.
 *        Splits a set of areas into a binary tree using octagon-based
 *        partitioning or K-Means++ clustering before bottom-up merging.
 */

#pragma once

#include <utility>
#include <vector>

#include "bound_skew_tree/component/Components.hh"

namespace icts::bst::detail {

class BoundSkewTreeImpl;
struct KMeansConfig;

class BinaryTopology
{
 public:
  explicit BinaryTopology(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~BinaryTopology() = default;
  BinaryTopology(const BinaryTopology&) = delete;
  auto operator=(const BinaryTopology&) -> BinaryTopology& = delete;

  auto biPartition() -> void;
  auto biCluster() -> void;

  static auto kMeansPlus(const std::vector<Area*>& areas, const KMeansConfig& config) -> std::vector<std::vector<Area*>>;
  static auto calcAreasCenter(const std::vector<Area*>& areas) -> Point;

 private:
  auto buildBiPartitionTree(const std::vector<Area*>& areas) -> Area*;
  auto buildBiClusterTree(const std::vector<Area*>& areas) -> Area*;
  static auto octagonDivide(std::vector<Area*>& areas) -> std::pair<std::vector<Area*>, std::vector<Area*>>;
  static auto calcOctagon(const std::vector<Area*>& areas) -> std::vector<Point>;
  static auto areaOnOctagonBound(const std::vector<Area*>& areas, const std::vector<Point>& octagon) -> std::vector<Area*>;

  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
