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
 * @file BoundaryConstraints.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief Resolves H-tree pattern search boundary constraints.
 */

#include "synthesis/htree/constraint/Constraint.hh"

#include <algorithm>

#include "ValueLattice.hh"
#include "characterization/Characterization.hh"

namespace icts::htree {

auto CoveringBoundaryIndex(double value, const UniformValueLattice& lattice) -> std::optional<unsigned>
{
  if (value <= 0.0 || !lattice.isValid()) {
    return std::nullopt;
  }
  if (value > lattice.maxValue() + kValueLatticeEpsilon) {
    return lattice.steps() + 1U;
  }
  return std::clamp(lattice.coveringIndex(value), 1U, lattice.steps());
}

auto ResolveBoundaryConstraints(const HTree::Config& config, const CharBuilder& char_builder) -> BoundaryConstraints
{
  BoundaryConstraints constraints;
  constraints.force_branch_buffer = config.force_branch_buffer;

  if (config.min_top_input_slew_ns.has_value() && *config.min_top_input_slew_ns >= 0.0) {
    constraints.min_top_input_slew_ns = config.min_top_input_slew_ns;
    if (*config.min_top_input_slew_ns > 0.0) {
      constraints.top_input_slew_covering_idx = CoveringBoundaryIndex(*config.min_top_input_slew_ns, char_builder.get_slew_lattice());
    }
  }

  return constraints;
}

auto HasBoundaryConstraints(const BoundaryConstraints& constraints) -> bool
{
  return constraints.top_input_slew_covering_idx.has_value();
}

auto ResolvePatternSearchBoundaryConstraints(const BoundaryConstraints& base_constraints, bool strict_root_boundary_closure)
    -> BoundaryConstraints
{
  auto constraints = base_constraints;
  if (strict_root_boundary_closure) {
    constraints.top_input_slew_covering_idx = std::nullopt;
  }
  return constraints;
}

}  // namespace icts::htree
