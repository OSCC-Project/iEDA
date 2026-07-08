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
 * @file WirelengthGrid.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree characterization wirelength grid resolution contracts.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "characterization/Characterization.hh"
#include "synthesis/htree/characterization/Characterization.hh"

namespace icts {
class Config;
class Tree;
}  // namespace icts

namespace icts::htree {

auto ToCharGridSourceName(CharGridSource source) -> const char*;
auto CountUniqueAlignedLengthBins(const std::vector<double>& requested_lengths_um, double length_step_um) -> unsigned;
auto CollectRequestedLevelLengthsUm(const Tree& topology, int32_t dbu_per_um) -> std::vector<double>;
auto ResolveCharacterizationGridPlan(const Config& config, const Tree& topology, int32_t dbu_per_um) -> CharacterizationGridPlan;
auto ResolveCharacterizationGridPlan(const Config& config, const std::vector<double>& requested_lengths_um) -> CharacterizationGridPlan;
auto ResolveCharacterizationGridPlan(const CharBuilder::Config& config, const std::vector<double>& requested_lengths_um)
    -> CharacterizationGridPlan;
auto ResolveDirectCharacterizationLengthIndices(const Tree& topology, const CharacterizationGridPlan& char_grid_plan, int32_t dbu_per_um)
    -> std::vector<unsigned>;
auto ResolveDirectCharacterizationLengthIndices(const std::vector<double>& requested_lengths_um,
                                                const CharacterizationGridPlan& char_grid_plan) -> std::vector<unsigned>;

}  // namespace icts::htree
