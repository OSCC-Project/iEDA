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
 * @file BoundSkewTree.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Bound-skew tree facade. Each public method forwards to the Pimpl
 *        implementation at `detail::BoundSkewTreeImpl`, which dispatches to
 *        the cooperating algorithm components.
 */

#include "bound_skew_tree/algorithm/BoundSkewTree.hh"

#include <memory>
#include <utility>

#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts {
enum class BSTRoutingRCPattern;
enum class BSTRoutingTopologyMode;
struct BSTRoutingConfig;
}  // namespace icts

namespace icts::bst {

BoundSkewTree::BoundSkewTree(std::vector<std::unique_ptr<Area>> load_areas, const BSTRoutingConfig& parameters,
                             const BSTRoutingTopologyMode& topology_mode)
    : _impl(std::make_unique<detail::BoundSkewTreeImpl>(std::move(load_areas), parameters, topology_mode))
{
}

BoundSkewTree::BoundSkewTree(std::vector<std::unique_ptr<Area>> owned_areas, Area* root, const BSTRoutingConfig& parameters)
    : _impl(std::make_unique<detail::BoundSkewTreeImpl>(std::move(owned_areas), root, parameters))
{
}

BoundSkewTree::~BoundSkewTree() = default;

auto BoundSkewTree::run() -> void
{
  _impl->run();
}

auto BoundSkewTree::get_root() const -> Area*
{
  return _impl->get_root();
}

auto BoundSkewTree::set_root_guide(const double& x_coord, const double& y_coord) -> void
{
  _impl->set_root_guide(x_coord, y_coord);
}

auto BoundSkewTree::set_rc_pattern(const BSTRoutingRCPattern& rc_pattern) -> void
{
  _impl->set_rc_pattern(rc_pattern);
}

}  // namespace icts::bst
