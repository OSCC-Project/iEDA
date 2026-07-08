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
 * @file BoundaryConstraints.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree pattern search boundary constraints derived from build config.
 */

#pragma once

#include <optional>

#include "synthesis/htree/HTree.hh"

namespace icts {

class CharBuilder;
class UniformValueLattice;

}  // namespace icts

namespace icts::htree {

struct BoundaryConstraints
{
  bool force_branch_buffer = false;
  std::optional<double> min_top_input_slew_ns = std::nullopt;
  std::optional<unsigned> top_input_slew_covering_idx = std::nullopt;
};

auto CoveringBoundaryIndex(double value, const UniformValueLattice& lattice) -> std::optional<unsigned>;
auto ResolveBoundaryConstraints(const HTree::Config& config, const CharBuilder& char_builder) -> BoundaryConstraints;
auto HasBoundaryConstraints(const BoundaryConstraints& constraints) -> bool;
auto ResolvePatternSearchBoundaryConstraints(const BoundaryConstraints& base_constraints, bool strict_root_boundary_closure)
    -> BoundaryConstraints;

}  // namespace icts::htree
