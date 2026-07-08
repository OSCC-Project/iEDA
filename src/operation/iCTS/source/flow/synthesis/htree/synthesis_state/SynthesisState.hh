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
 * @file SynthesisState.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-27
 * @brief Shared H-tree synthesis-state contract.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts {

class CharBuilder;

namespace htree {

enum class HTreeSynthesisStateStatus
{
  kReady,
  kCompleted,
  kFailed,
};

struct HTreeSynthesisState
{
  const HTree::Input* input = nullptr;
  const HTree::Config* config = nullptr;
  DiagnosticBuild result;

  std::optional<CharacterizationLibrary> local_char_library = std::nullopt;
  double char_length_step_um = 0.0;

  BoundaryConstraints base_boundary_constraints;
  BoundaryConstraints search_boundary_constraints;
  std::vector<HTree::LevelPlan> full_level_plans;
  std::vector<unsigned> depth_candidates;
  unsigned max_depth = 0U;

  std::optional<BufferPatternLibrary> segment_pattern_library = std::nullopt;
  HTreeFanoutPruningConfig fanout_pruning_config;
  RootDriverCompensationInput root_driver_compensation_input;
  SinkLoadRegionLegalityInput sink_load_region_input;
  std::string root_driver_clock_period_source;
  bool strict_root_boundary_closure = false;

  auto charLibrary() -> CharacterizationLibrary&;
  auto charBuilder() const -> const CharBuilder&;
  auto segmentPatterns() -> BufferPatternLibrary&;
};

struct HTreeSynthesisStateBuild
{
  HTreeSynthesisStateStatus status = HTreeSynthesisStateStatus::kFailed;
  std::string failure_reason;
  HTreeSynthesisState state;
};

auto AssembleHTreeSynthesisState(const HTree::Input& input, const HTree::Config& config, SchemaWriter::StageScope& build_stage)
    -> HTreeSynthesisStateBuild;

}  // namespace htree
}  // namespace icts
