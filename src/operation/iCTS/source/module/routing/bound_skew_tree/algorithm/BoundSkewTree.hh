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
 * @file BoundSkewTree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-08
 * @brief Bound-skew tree construction facade. The algorithm internals live
 *        behind a Pimpl boundary at `detail::BoundSkewTreeImpl`.
 */
#pragma once

#include <memory>
#include <vector>

namespace icts {
enum class BSTRoutingRCPattern;
enum class BSTRoutingTopologyMode;
struct BSTRoutingConfig;
}  // namespace icts

namespace icts::bst {
class Area;
}  // namespace icts::bst

namespace icts::bst::detail {
class BoundSkewTreeImpl;
}  // namespace icts::bst::detail

namespace icts::bst {

class BoundSkewTree
{
 public:
  static constexpr double kHalfFactor = 0.5;

  BoundSkewTree(std::vector<std::unique_ptr<Area>> load_areas, const BSTRoutingConfig& parameters,
                const BSTRoutingTopologyMode& topology_mode);
  BoundSkewTree(std::vector<std::unique_ptr<Area>> owned_areas, Area* root, const BSTRoutingConfig& parameters);
  BoundSkewTree(const BoundSkewTree&) = delete;
  BoundSkewTree(BoundSkewTree&&) = delete;
  auto operator=(const BoundSkewTree&) -> BoundSkewTree& = delete;
  auto operator=(BoundSkewTree&&) -> BoundSkewTree& = delete;
  ~BoundSkewTree();

  auto run() -> void;
  auto get_root() const -> Area*;
  auto set_root_guide(const double& x_coord, const double& y_coord) -> void;
  auto set_rc_pattern(const BSTRoutingRCPattern& rc_pattern) -> void;

 private:
  std::unique_ptr<detail::BoundSkewTreeImpl> _impl;
};

}  // namespace icts::bst
