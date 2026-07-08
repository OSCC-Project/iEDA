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
 * @file LevelPlan.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree topology level planning contracts.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "synthesis/htree/HTree.hh"

namespace icts {
class Tree;
}  // namespace icts

namespace icts::htree {

auto BuildLevelPlans(const Tree& topology, double length_step_um, int32_t dbu_per_um) -> std::vector<HTree::LevelPlan>;
auto MakeCandidateLevelPlans(const std::vector<HTree::LevelPlan>& full_level_plans, unsigned depth) -> std::vector<HTree::LevelPlan>;
auto CountCandidateLeafNodes(const Tree& topology, unsigned depth) -> std::size_t;
auto ResolveDepthCandidates(unsigned max_depth, const HTree::Config& config) -> std::vector<unsigned>;

}  // namespace icts::htree
