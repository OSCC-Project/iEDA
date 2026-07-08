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
 * @file HTreeRealTechScenario.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Compiled helpers for real-tech H-tree smoke tests.
 */

#include "flow/synthesis/htree/HTreeRealTechScenario.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

#include "Clock.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Tree.hh"
#include "flow/synthesis/htree/HTreeArtifactWriter.hh"
#include "flow/synthesis/htree/HTreeBuildObservation.hh"
#include "synthesis/htree/HTree.hh"
#if defined(ICTS_ENABLE_SLOW_REALTECH_REGRESSION) && ICTS_ENABLE_SLOW_REALTECH_REGRESSION
#include "Point.hh"
#endif
#include "database/config/Config.hh"
#include "database/design/Design.hh"
#include "setup/clock_data/ClockDataRead.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"

namespace icts_test {

auto ReadEnvFlag(std::string_view env_name) -> bool
{
  const char* raw_value = std::getenv(std::string(env_name).c_str());
  if (raw_value == nullptr) {
    return false;
  }

  const std::string value = raw_value;
  return !(value.empty() || value == "0" || value == "false" || value == "FALSE" || value == "False");
}

auto FormatArm9ExperimentReport(std::string_view scenario_name, const std::string& clock_name, std::size_t load_count,
                                bool omit_wirelength_unit, const std::vector<Arm9ExperimentRecord>& records) -> std::string
{
  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(6);
  report_stream << "scenario=" << scenario_name << "\n";
  report_stream << "clock_name=" << clock_name << "\n";
  report_stream << "load_count=" << load_count << "\n";
  report_stream << "omit_wirelength_unit=" << (omit_wirelength_unit ? "true" : "false") << "\n";
  report_stream << "runtime_budget_s=" << kArm9ExperimentRuntimeBudgetS << "\n";
  report_stream << "columns=iter,step,runtime_s,success,frontier_count,selected_depth,best_pattern_id,best_delay_ns,best_power_w,"
                   "char_wirelength_unit_um,char_wirelength_iterations,char_grid_adapted,used_boundary_relaxation,failure_reason\n";
  for (const auto& record : records) {
    report_stream << record.wirelength_iterations << "," << record.slew_cap_steps << "," << record.runtime_s << ","
                  << (record.success ? "true" : "false") << "," << record.final_frontier_count << "," << record.selected_depth << ","
                  << record.best_pattern_id << "," << record.best_delay_ns << "," << record.best_power_w << ","
                  << record.char_wirelength_unit_um << "," << record.char_wirelength_iterations << ","
                  << (record.char_grid_adapted ? "true" : "false") << "," << (record.used_boundary_relaxation ? "true" : "false") << ","
                  << record.failure_reason << "\n";
  }
  return report_stream.str();
}

auto SampleLoadsForSmoke(const std::vector<icts::Pin*>& loads, std::size_t max_count) -> std::vector<icts::Pin*>
{
  if (loads.size() <= max_count) {
    return loads;
  }

  std::vector<icts::Pin*> sampled_loads;
  sampled_loads.reserve(max_count);
  for (std::size_t sample_index = 0; sample_index < max_count; ++sample_index) {
    const std::size_t source_index = sample_index * loads.size() / max_count;
    sampled_loads.push_back(loads.at(source_index));
  }
  return sampled_loads;
}

auto ConnectRootNetForHTreeTest(icts::Net& root_net, icts::Pin& root_driver, const std::vector<icts::Pin*>& loads) -> void
{
  root_net.set_driver(&root_driver);
  root_driver.set_net(&root_net);
  root_net.set_loads(loads);
  for (auto* load : loads) {
    if (load != nullptr) {
      load->set_net(&root_net);
    }
  }
}

auto SelectLargestRealClockLoads(std::size_t max_count) -> std::optional<RealClockLoadSelection>
{
  icts_test::runtime::CurrentRuntime().design.reset();
  auto& runtime = icts_test::runtime::CurrentRuntime();
  if (!icts::ClockDataRead::read(icts::ClockDataReadInput{
          .config = &runtime.config,
          .design = &runtime.design,
          .wrapper = &runtime.wrapper,
          .reporter = &runtime.reporter,
      })) {
    return std::nullopt;
  }

  RealClockLoadSelection best_selection;
  std::size_t best_source_load_count = 0U;
  for (auto* clock : icts_test::runtime::CurrentRuntime().design.get_clocks()) {
    if (clock == nullptr || clock->get_loads().size() < 2U) {
      continue;
    }
    if (clock->get_loads().size() <= best_source_load_count) {
      continue;
    }

    best_selection.clock_name = clock->get_clock_name() + ":" + clock->get_clock_net_name();
    best_selection.loads = SampleLoadsForSmoke(clock->get_loads(), max_count);
    best_source_load_count = clock->get_loads().size();
  }

  if (best_selection.loads.size() < 2U) {
    return std::nullopt;
  }

  return best_selection;
}

auto CountPinsWithRealContext(const std::vector<icts::Pin*>& loads) -> std::size_t
{
  return static_cast<std::size_t>(std::ranges::count_if(loads, [](const icts::Pin* pin) -> bool {
    return pin != nullptr && pin->get_inst() != nullptr && pin->get_net() != nullptr && !pin->get_name().empty()
           && !pin->get_inst()->get_name().empty();
  }));
}

namespace {

auto ResolveRoutingLayerForHTreeTest(const icts::Config& config) -> int
{
  const auto& routing_layers = config.get_routing_layers();
  return routing_layers.empty() ? 0 : static_cast<int>(routing_layers.front());
}

auto ResolveWireWidthForHTreeTest(const icts::Config& config) -> std::optional<double>
{
  const double wire_width_um = config.get_wire_width();
  return wire_width_um > 0.0 ? std::optional<double>(wire_width_um) : std::nullopt;
}

}  // namespace

auto MakeExplicitHTreeInput(icts::Net& root_net) -> icts::HTree::Input
{
  auto& runtime = icts_test::runtime::CurrentRuntime();
  return icts::HTree::Input{
      .root_net = &root_net,
      .design = &runtime.design,
      .wrapper = &runtime.wrapper,
      .reporter = &runtime.reporter,
      .characterization_input = icts::CharacterizationLibrary::buildRuntimeInput(icts::CharacterizationRuntimeInput{
          .config = &runtime.config,
          .wrapper = &runtime.wrapper,
          .fast_sta = &runtime.fast_sta,
          .reporter = &runtime.reporter,
      }),
      .characterization_config = icts::CharacterizationLibrary::buildRuntimeConfig(runtime.config),
      .additional_characterization_lengths_um = {},
  };
}

auto MakeExplicitHTreeConfig(std::optional<bool> force_branch_buffer, std::optional<double> min_top_input_slew_ns) -> icts::HTree::Config
{
  auto& runtime = icts_test::runtime::CurrentRuntime();
  return icts::HTree::Config{
      .force_branch_buffer = force_branch_buffer.value_or(runtime.config.is_force_branch_buffer()),
      .min_top_input_slew_ns = min_top_input_slew_ns,
      .depth_explore_window = std::max(1U, runtime.config.get_htree_depth_explore_window()),
      .topology_tolerance = runtime.config.get_htree_topology_tolerance(),
      .max_fanout = runtime.config.get_max_fanout(),
      .has_max_cap = runtime.config.has_max_cap(),
      .max_cap_pf = runtime.config.has_max_cap() ? runtime.config.get_max_cap() : 0.0,
      .enable_root_driver_sizing = true,
      .allow_boundary_relaxation = true,
      .enable_analytical_solver = runtime.config.is_enable_analytical_htree(),
      .routing_layer = ResolveRoutingLayerForHTreeTest(runtime.config),
      .wire_width_um = ResolveWireWidthForHTreeTest(runtime.config),
  };
}

auto CollectLeafLoads(const icts::Tree& topology) -> std::unordered_set<icts::Pin*>
{
  std::unordered_set<icts::Pin*> leaf_loads;
  const auto levels = topology.levels();
  if (levels.empty()) {
    return leaf_loads;
  }

  for (const auto node_id : levels.back()) {
    const auto* node = topology.get_node(node_id);
    if (node == nullptr || !node->isLeaf()) {
      continue;
    }
    for (auto* load : node->get_loads()) {
      if (load != nullptr) {
        leaf_loads.insert(load);
      }
    }
  }

  return leaf_loads;
}

auto AssertNoSingleLoadExternalLeafBuffer(const icts::htree::DiagnosticBuild& result) -> void
{
  std::unordered_set<const icts::Inst*> inserted_insts;
  inserted_insts.reserve(result.output.inserted_insts.size());
  for (const auto& inst_owner : result.output.inserted_insts) {
    inserted_insts.insert(inst_owner.get());
  }

  for (const auto& inst_owner : result.output.inserted_insts) {
    const auto* inst = inst_owner.get();
    if (inst == nullptr || !inst->is_buffer()) {
      continue;
    }

    const auto* output_pin = inst->findDriverPin();
    ASSERT_NE(output_pin, nullptr);
    const auto* output_net = output_pin->get_net();
    if (output_net == nullptr || output_net->get_loads().size() != 1U || output_net->get_loads().front() == nullptr) {
      continue;
    }

    const auto* downstream_load = output_net->get_loads().front();
    const auto* downstream_inst = downstream_load->get_inst();
    const bool drives_internal_htree_inst = downstream_inst != nullptr && inserted_insts.contains(downstream_inst);
    EXPECT_TRUE(drives_internal_htree_inst) << "Expected leaf single-load buffer pruning to remove " << inst->get_name()
                                            << " but it still drives " << downstream_load->get_name();
  }
}

auto ReadTextFile(const std::filesystem::path& path) -> std::string
{
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return {};
  }

  std::ostringstream content_stream;
  content_stream << input_stream.rdbuf();
  return content_stream.str();
}

auto AssertDepthCandidateCoverage(const icts::htree::DiagnosticBuild& result) -> void
{
  const auto observation = htree::ObserveHTreeBuild(result);
  ASSERT_GT(observation.depth_candidate_count, 0U);
  ASSERT_TRUE(observation.has_selected_depth);

  const auto topology_levels = result.output.topology.levels();
  ASSERT_GT(topology_levels.size(), 1U);
  const auto max_depth = static_cast<unsigned>(topology_levels.size() - 1U);
  EXPECT_EQ(observation.depth_candidate_count,
            std::min<std::size_t>(icts_test::runtime::CurrentRuntime().config.get_htree_depth_explore_window(), max_depth));

  EXPECT_EQ(observation.selected_depth, result.summary.selected_depth.value_or(0U));
  EXPECT_EQ(observation.selected_depth, observation.selected_level_count);
  EXPECT_TRUE(observation.success);
  EXPECT_GT(observation.selected_final_frontier_count, 0U);
}

auto AssertSelectedHTreeLoadDistribution(const icts::htree::DiagnosticBuild& result) -> void
{
  const auto observation = htree::ObserveHTreeBuild(result);
  ASSERT_GT(observation.htree_load_group_count, 0U);
  EXPECT_LE(observation.htree_load_cap_min_pf, observation.htree_load_cap_mean_pf);
  EXPECT_LE(observation.htree_load_cap_min_pf, observation.htree_load_cap_median_pf);
  EXPECT_LE(observation.htree_load_cap_mean_pf, observation.htree_load_cap_max_pf);
  EXPECT_LE(observation.htree_load_cap_median_pf, observation.htree_load_cap_max_pf);
}

auto WriteAndAssertHTreeArtifacts(const htree::HTreeArtifactPaths& artifact_paths, const std::string& scenario_name,
                                  const std::string& clock_name, const std::vector<icts::Pin*>& loads,
                                  const icts::htree::DiagnosticBuild& result) -> void
{
  ASSERT_FALSE(artifact_paths.output_dir.empty());
  EXPECT_TRUE(htree::WriteHTreeArtifacts(artifact_paths, scenario_name, clock_name, loads, result));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.cts_log));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.topology_svg));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.materialized_svg));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.pareto_svg));
  EXPECT_TRUE(std::filesystem::exists(artifact_paths.report_log));
}

#if ICTS_ENABLE_SLOW_REALTECH_REGRESSION
auto AssertBranchBufferMaterialization(const icts::htree::DiagnosticBuild& result) -> void
{
  ASSERT_TRUE(result.summary.success);
  ASSERT_FALSE(result.output.levels.empty());
  ASSERT_TRUE(result.diagnostics.force_branch_buffer);

  std::vector<icts::Point<int>> required_terminal_positions;
  const auto topology_levels = result.output.topology.levels();
  for (std::size_t level_index = 0; level_index < result.output.levels.size(); ++level_index) {
    const auto& level = result.output.levels.at(level_index);
    EXPECT_TRUE(level.selected_has_terminal_branch_buffer);

    if (!level.is_leaf_level || level_index + 1U >= topology_levels.size()) {
      continue;
    }
    for (const auto node_id : topology_levels.at(level_index + 1U)) {
      const auto* node = result.output.topology.get_node(node_id);
      if (node == nullptr || node->get_loads().empty()) {
        continue;
      }
      required_terminal_positions.push_back(node->get_position());
    }
  }

  for (const auto& expected_position : required_terminal_positions) {
    const bool has_terminal_inst = std::ranges::any_of(result.output.inserted_insts, [&expected_position](const auto& inst_owner) -> bool {
      const auto* inst = inst_owner.get();
      return inst != nullptr && inst->get_location() == expected_position;
    });
    EXPECT_TRUE(has_terminal_inst);
  }
}

#endif

}  // namespace icts_test
