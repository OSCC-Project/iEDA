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
 * @file HTreeRealTechScenario.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech H-tree smoke fixture and assertion helpers.
 */

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "Pin.hh"
#include "flow/synthesis/htree/HTree.hh"
#include "flow/synthesis/htree/diagnostic/HTreeDiagnostic.hh"

namespace icts {
class Net;
class Tree;
}  // namespace icts

namespace icts_test::htree {
struct HTreeArtifactPaths;
}  // namespace icts_test::htree

namespace icts_test {

#ifndef ICTS_ENABLE_SLOW_REALTECH_REGRESSION
#define ICTS_ENABLE_SLOW_REALTECH_REGRESSION 0
#endif

inline constexpr std::size_t kMaxRealClockLoadCount = 64U;
inline constexpr double kHTreeSmokeMaxSlewNs = 0.05;
inline constexpr double kHTreeSmokeMaxCapPf = 0.15;
inline constexpr double kArm9ExperimentRuntimeBudgetS = 600.0;
inline constexpr std::string_view kRunArm9ExperimentEnv = "ICTS_RUN_ARM9_HTREE_MATRIX";
inline constexpr std::string_view kRunArm9ExperimentAutoUnitEnv = "ICTS_RUN_ARM9_HTREE_MATRIX_AUTO_UNIT";
inline constexpr std::string_view kArm9ExperimentScenario = "htree_arm9_full_sink_matrix";
inline constexpr std::string_view kArm9ExperimentAutoUnitScenario = "htree_arm9_full_sink_matrix_auto_unit";
inline constexpr std::array<unsigned, 4> kArm9ExperimentIterations = {2U, 3U, 4U, 5U};
inline constexpr std::array<unsigned, 2> kArm9ExperimentSteps = {10U, 15U};

struct RealClockLoadSelection
{
  std::string clock_name;
  std::vector<icts::Pin*> loads;
};

struct Arm9ExperimentRecord
{
  unsigned wirelength_iterations = 0U;
  unsigned slew_cap_steps = 0U;
  double runtime_s = 0.0;
  bool success = false;
  std::size_t load_count = 0U;
  std::size_t final_frontier_count = 0U;
  unsigned selected_depth = 0U;
  unsigned best_pattern_id = 0U;
  double best_delay_ns = 0.0;
  double best_power_w = 0.0;
  double char_wirelength_unit_um = 0.0;
  unsigned char_wirelength_iterations = 0U;
  bool char_grid_adapted = false;
  bool used_boundary_relaxation = false;
  std::string failure_reason;
};

struct Arm9ExperimentMatrixRunResult
{
  bool skipped = false;
  std::string skip_reason;
  RealClockLoadSelection selection;
  std::vector<Arm9ExperimentRecord> records;
  std::vector<std::string> failure_messages;
  bool report_written = false;
};

auto ReadEnvFlag(std::string_view env_name) -> bool;
auto FormatArm9ExperimentReport(std::string_view scenario_name, const std::string& clock_name, std::size_t load_count,
                                bool omit_wirelength_unit, const std::vector<Arm9ExperimentRecord>& records) -> std::string;
auto SampleLoadsForSmoke(const std::vector<icts::Pin*>& loads, std::size_t max_count) -> std::vector<icts::Pin*>;
auto ConnectRootNetForHTreeTest(icts::Net& root_net, icts::Pin& root_driver, const std::vector<icts::Pin*>& loads) -> void;
auto SelectLargestRealClockLoads(std::size_t max_count) -> std::optional<RealClockLoadSelection>;
auto CountPinsWithRealContext(const std::vector<icts::Pin*>& loads) -> std::size_t;
auto MakeExplicitHTreeInput(icts::Net& root_net) -> icts::HTree::Input;
auto MakeExplicitHTreeConfig(std::optional<bool> force_branch_buffer = std::nullopt,
                             std::optional<double> min_top_input_slew_ns = std::nullopt) -> icts::HTree::Config;
auto CollectLeafLoads(const icts::Tree& topology) -> std::unordered_set<icts::Pin*>;
auto AssertNoSingleLoadExternalLeafBuffer(const icts::htree::DiagnosticBuild& result) -> void;
auto ReadTextFile(const std::filesystem::path& path) -> std::string;
auto AssertDepthCandidateCoverage(const icts::htree::DiagnosticBuild& result) -> void;
auto AssertSelectedHTreeLoadDistribution(const icts::htree::DiagnosticBuild& result) -> void;
auto WriteAndAssertHTreeArtifacts(const htree::HTreeArtifactPaths& artifact_paths, const std::string& scenario_name,
                                  const std::string& clock_name, const std::vector<icts::Pin*>& loads,
                                  const icts::htree::DiagnosticBuild& result) -> void;
auto EvaluateArm9FullSinkExperimentMatrix(bool omit_wirelength_unit) -> Arm9ExperimentMatrixRunResult;

#if ICTS_ENABLE_SLOW_REALTECH_REGRESSION
auto AssertBranchBufferMaterialization(const icts::htree::DiagnosticBuild& result) -> void;
#endif

}  // namespace icts_test
