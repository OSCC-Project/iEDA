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
 * @file SegmentPruning.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree segment frontier synthesis contracts.
 */

#pragma once

#include <vector>

#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/segment_pruning/SegmentFrontierCatalog.hh"

namespace icts {
class SegmentChar;
}  // namespace icts

namespace icts::htree {

struct BoundaryConstraints;
struct BufferPatternLibrary;

auto CollectRequiredLengthIndices(const std::vector<HTree::LevelPlan>& levels) -> std::vector<unsigned>;
auto ResolveRequiredSegmentFrontiers(std::vector<unsigned> required_length_indices, const BoundaryConstraints& boundary_constraints)
    -> RequiredSegmentFrontiers;
auto SynthesizeSegmentFrontiers(const std::vector<SegmentChar>& base_segment_chars, BufferPatternLibrary& pattern_library,
                                const RequiredSegmentFrontiers& required_frontiers) -> SegmentFrontierCatalog;

}  // namespace icts::htree
