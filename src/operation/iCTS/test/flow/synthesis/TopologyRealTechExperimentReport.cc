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
 * @file TopologyRealTechExperimentReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech clock selection and matrix report helpers for Topology tests.
 */

#include <compare>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Pin.hh"
#include "TopologyRealTechScenario.hh"
#include "Tree.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "database/design/Net.hh"
#include "database/io/Wrapper.hh"
#include "synthesis/topology/Topology.hh"

namespace icts_test::synthesis_realtech_smoke {
namespace {

auto ResolveBufferDriveCap(const std::string& cell_master) -> double
{
  double drive_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapLimit(cell_master);
  if (drive_cap_pf <= 0.0) {
    drive_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapTableAxisMax(cell_master);
  }
  return drive_cap_pf;
}

}  // namespace

auto FormatTopologyExperimentReport(std::string_view scenario_name, const RealClockSelection& selection, bool omit_wirelength_unit,
                                    const std::vector<TopologyExperimentRecord>& records) -> std::string
{
  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(6);
  report_stream << "scenario=" << scenario_name << "\n";
  report_stream << "clock_name=" << selection.clock_name << "\n";
  report_stream << "clock_net=" << selection.net_name << "\n";
  report_stream << "sink_count=" << selection.sinks.size() << "\n";
  report_stream << "source_net_load_count=" << selection.source_net_load_count << "\n";
  report_stream << "def_clock_net=" << (selection.is_def_clock_net ? "true" : "false") << "\n";
  report_stream << "clock_like_load_pin_count=" << selection.clock_like_load_pin_count << "\n";
  report_stream << "sink_clustering_enabled=false\n";
  report_stream << "omit_wirelength_unit=" << (omit_wirelength_unit ? "true" : "false") << "\n";
  report_stream << "runtime_budget_s=" << kBpBeTopSynthesisRuntimeBudgetS << "\n";
  report_stream << "columns=iter,step,runtime_s,success,selected_depth,best_pattern_id,best_delay_ns,best_power_w,"
                   "used_boundary_relaxation,failure_reason\n";
  for (const auto& record : records) {
    report_stream << record.wirelength_iterations << "," << record.slew_cap_steps << "," << record.runtime_s << ","
                  << (record.success ? "true" : "false") << "," << record.selected_depth << "," << record.best_pattern_id << ","
                  << record.best_delay_ns << "," << record.best_power_w << "," << (record.used_boundary_relaxation ? "true" : "false")
                  << "," << record.failure_reason << "\n";
  }
  return report_stream.str();
}

auto FormatTopologyToleranceComparisonReport(std::string_view scenario_name, const RealClockSelection& selection,
                                             const TopologyToleranceComparisonResult& comparison) -> std::string
{
  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(6);
  report_stream << "scenario=" << scenario_name << "\n";
  report_stream << "clock_name=" << selection.clock_name << "\n";
  report_stream << "clock_net=" << selection.net_name << "\n";
  report_stream << "sink_count=" << selection.sinks.size() << "\n";
  report_stream << "sink_clustering_enabled=false\n";
  report_stream << "char_policy=per_level_baseline_segment_length\n";
  report_stream << "sta_skew_method=routed_rc_tree_arrival_max_minus_min\n";
  report_stream << "wirelength_skew_method=routed_path_wirelength_max_minus_min\n";
  report_stream << "paired_delta_mean_dbu=" << comparison.mean_distance_delta_dbu << "\n";
  report_stream << "paired_delta_max_abs_dbu=" << comparison.max_abs_distance_delta_dbu << "\n";
  report_stream << "improved_load_count=" << comparison.improved_load_count << "\n";
  report_stream << "worsened_load_count=" << comparison.worsened_load_count << "\n";
  report_stream << "unchanged_load_count=" << comparison.unchanged_load_count << "\n";
  report_stream << "columns=tolerance,runtime_s,success,sink_count,leaf_load_distance_mean_dbu,leaf_load_distance_max_dbu,"
                   "sta_arrival_skew_ns,wirelength_skew_dbu,sta_sink_count,selected_char_delay_ns,failure_reason\n";
  for (const auto& record : comparison.records) {
    report_stream << record.htree_topology_tolerance << "," << record.runtime_s << "," << (record.success ? "true" : "false") << ","
                  << record.sink_count << "," << record.leaf_load_distance_mean_dbu << "," << record.leaf_load_distance_max_dbu << ","
                  << record.sta_arrival_skew_ns << "," << record.wirelength_skew_dbu << "," << record.sta_sink_count << ","
                  << record.selected_char_delay_ns << "," << record.failure_reason << "\n";
  }
  return report_stream.str();
}

auto WriteTopologyMatrixReport(std::string_view scenario_name, const std::string& file_name, const std::string& content) -> bool
{
  const auto output_dir = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "flow" / "synthesis"
                                                            / common::io::SanitizeOutputName(std::string(scenario_name)));
  if (output_dir.empty()) {
    return false;
  }
  return common::io::WriteTextLog(output_dir / file_name, content);
}

auto SelectLargestRealClock(std::size_t max_count, std::size_t min_required_load_count) -> std::optional<RealClockSelection>
{
  return common::realtech::SelectLargestDefClock(max_count, min_required_load_count);
}

auto SetEnableSinkClustering(icts::Topology::Config& config, bool enabled) -> void
{
  config.enable_sink_clustering = enabled;
}

auto BuildTopology(icts::Net& root_net, const icts::Topology::Config& config) -> icts::Topology::Build
{
  auto& runtime = icts_test::runtime::CurrentRuntime();
  icts::Topology::Input input{
      .config = &runtime.config,
      .design = &runtime.design,
      .wrapper = &runtime.wrapper,
      .fast_sta = &runtime.fast_sta,
      .reporter = &runtime.reporter,
      .root_net = &root_net,
      .object_name_prefix = {},
  };
  return icts::Topology::build(input, config);
}

auto ConnectRootNet(icts::Net& root_net, icts::Pin* source, const std::vector<icts::Pin*>& sinks) -> void
{
  root_net.set_driver(source);
  if (source != nullptr) {
    source->set_net(&root_net);
  }

  root_net.set_loads({});
  for (auto* sink : sinks) {
    if (sink == nullptr) {
      continue;
    }
    root_net.add_load(sink);
    sink->set_net(&root_net);
  }
}

auto ResolveExpectedMinClusterBufferMaster() -> std::optional<std::string>
{
  std::optional<std::string> expected_master = std::nullopt;
  double best_drive_cap_pf = std::numeric_limits<double>::infinity();
  for (const auto& cell_master : icts_test::runtime::CurrentRuntime().config.get_buffer_types()) {
    if (cell_master.empty()) {
      continue;
    }

    const auto [input_pin, output_pin] = icts_test::runtime::CurrentRuntime().wrapper.queryBufferPorts(cell_master);
    if (input_pin.empty() || output_pin.empty()) {
      continue;
    }

    const double drive_cap_pf = ResolveBufferDriveCap(cell_master);
    if (drive_cap_pf <= 0.0) {
      continue;
    }

    if (!expected_master.has_value() || drive_cap_pf < best_drive_cap_pf
        || (drive_cap_pf == best_drive_cap_pf && cell_master < *expected_master)) {
      expected_master = cell_master;
      best_drive_cap_pf = drive_cap_pf;
    }
  }

  return expected_master;
}

auto CalcFloorPowerOfTwo(std::size_t value) -> std::size_t
{
  if (value == 0U) {
    return 0U;
  }

  std::size_t power = 1U;
  while ((power << 1U) <= value) {
    power <<= 1U;
  }
  return power;
}

auto CountTopologyLeafNodes(const icts::Tree& topology) -> std::size_t
{
  const auto levels = topology.levels();
  if (levels.empty()) {
    return 0U;
  }
  return levels.back().size();
}

}  // namespace icts_test::synthesis_realtech_smoke
