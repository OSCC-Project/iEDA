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
 * @file Characterization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief H-tree characterization grid config and flow summary contracts.
 */

#pragma once

#include <cstdint>
#include <string>

#include "characterization/Characterization.hh"
#include "synthesis/htree/HTree.hh"

namespace icts {
class CharacterizationLibrary;
class Tree;
}  // namespace icts

namespace icts::htree {

struct DiagnosticBuild;

enum class CharGridSource
{
  kNone,
  kRuntimeConfig,
  kAutoDerived
};

struct CharacterizationGridPlan
{
  double wirelength_unit_um = 0.0;
  unsigned wirelength_iterations = 0U;
  unsigned configured_wirelength_iterations = 0U;
  unsigned required_covering_iterations = 0U;
  unsigned unique_level_bins = 0U;
  double configured_wirelength_unit_um = 0.0;
  double auto_derived_wirelength_unit_um = 0.0;
  unsigned requested_level_lengths = 0U;
  bool configured_wirelength_missing = false;
  bool configured_grid_collapsed = false;
  bool adapted = false;
  CharGridSource source = CharGridSource::kNone;
};

struct CharacterizationSummary
{
  bool success = false;
  std::string failure_reason;
  double length_step_um = 0.0;
};

auto RunCharacterizationFlow(const Tree& topology, int32_t dbu_per_um, const CharBuilder::Input& base_char_input,
                             const CharBuilder::Config& base_char_config, htree::DiagnosticBuild& result,
                             CharacterizationLibrary& char_library, const HTree::Input& input, const HTree::Config& config)
    -> CharacterizationSummary;

}  // namespace icts::htree
