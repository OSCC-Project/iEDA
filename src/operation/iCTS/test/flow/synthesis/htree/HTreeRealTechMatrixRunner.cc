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
 * @file HTreeRealTechMatrixRunner.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief ARM9 real-tech matrix runner for HTree tests.
 */

#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Net.hh"
#include "Pin.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "flow/synthesis/htree/HTree.hh"
#include "flow/synthesis/htree/HTreeBuildObservation.hh"
#include "flow/synthesis/htree/HTreeRealTechScenario.hh"
#include "flow/synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"

namespace icts_test {
namespace {

namespace common_realtech = common::realtech;
namespace realtech_fixture = characterization::realtech;

auto MakeSkipResult(const std::string& reason) -> Arm9ExperimentMatrixRunResult
{
  Arm9ExperimentMatrixRunResult result;
  result.skipped = true;
  result.skip_reason = reason;
  return result;
}

auto MakeCasePrefix(unsigned wirelength_iterations, unsigned slew_cap_steps) -> std::string
{
  return "iter=" + std::to_string(wirelength_iterations) + ", step=" + std::to_string(slew_cap_steps) + ": ";
}

auto ResolveArm9MatrixEnvName(bool omit_wirelength_unit) -> std::string_view
{
  return omit_wirelength_unit ? kRunArm9ExperimentAutoUnitEnv : kRunArm9ExperimentEnv;
}

auto ResolveArm9MatrixScenarioName(bool omit_wirelength_unit) -> std::string_view
{
  return omit_wirelength_unit ? kArm9ExperimentAutoUnitScenario : kArm9ExperimentScenario;
}

auto MakeArm9CaseScenarioName(bool omit_wirelength_unit, unsigned wirelength_iterations, unsigned slew_cap_steps) -> std::string
{
  std::ostringstream scenario_name_stream;
  scenario_name_stream << "htree_arm9_full_sink";
  if (omit_wirelength_unit) {
    scenario_name_stream << "_auto_unit";
  }
  scenario_name_stream << "_iter" << wirelength_iterations << "_step" << slew_cap_steps;
  return scenario_name_stream.str();
}

auto MakeArm9ExperimentRecord(unsigned wirelength_iterations, unsigned slew_cap_steps, double runtime_s,
                              const icts::htree::DiagnosticBuild& result, std::size_t load_count) -> Arm9ExperimentRecord
{
  const auto observation = htree::ObserveHTreeBuild(result);
  Arm9ExperimentRecord record{
      .wirelength_iterations = wirelength_iterations,
      .slew_cap_steps = slew_cap_steps,
      .runtime_s = runtime_s,
      .success = result.summary.success,
      .load_count = load_count,
      .final_frontier_count = observation.selected_final_frontier_count,
      .selected_depth = observation.selected_depth,
      .best_pattern_id = observation.best_pattern_id,
      .best_delay_ns = observation.best_delay_ns,
      .best_power_w = observation.best_power_w,
      .char_wirelength_unit_um = result.diagnostics.char_wirelength_unit_um,
      .char_wirelength_iterations = result.diagnostics.char_wirelength_iterations,
      .char_grid_adapted = result.diagnostics.char_grid_adapted,
      .used_boundary_relaxation = observation.used_boundary_relaxation,
      .failure_reason = result.summary.failure_reason,
  };

  return record;
}

auto AppendArm9CaseFailures(unsigned wirelength_iterations, unsigned slew_cap_steps, bool omit_wirelength_unit, double runtime_s,
                            const icts::htree::DiagnosticBuild& result, const Arm9ExperimentRecord& record,
                            std::vector<std::string>& failure_messages) -> void
{
  const std::string prefix = MakeCasePrefix(wirelength_iterations, slew_cap_steps);
  if (!result.summary.success) {
    failure_messages.push_back(prefix + "failure_reason=" + result.summary.failure_reason);
  }
  if (runtime_s > kArm9ExperimentRuntimeBudgetS) {
    failure_messages.push_back(prefix + "runtime_s=" + std::to_string(runtime_s) + " exceeds budget");
  }
  if (record.final_frontier_count == 0U) {
    failure_messages.push_back(prefix + "final frontier count is zero");
  }
  if (!result.output.best_char.has_value()) {
    failure_messages.push_back(prefix + "best htree char is missing");
  }
  if (omit_wirelength_unit && record.char_wirelength_unit_um <= 0.0) {
    failure_messages.push_back(prefix + "auto-derived char wirelength unit is not positive");
  }
  if (omit_wirelength_unit && record.char_wirelength_iterations > wirelength_iterations) {
    failure_messages.push_back(prefix + "char wirelength iterations exceed requested iteration count");
  }
}

auto ValidateSelectedClock(const RealClockLoadSelection& selected_clock, std::vector<std::string>& failure_messages) -> bool
{
  if (selected_clock.loads.size() < 2U) {
    failure_messages.emplace_back("Selected clock has fewer than two loads.");
    return false;
  }

  const std::size_t real_context_count = CountPinsWithRealContext(selected_clock.loads);
  if (real_context_count != selected_clock.loads.size()) {
    failure_messages.push_back("Selected clock loads do not carry complete DEF/CTS instance context: " + selected_clock.clock_name);
    return false;
  }
  return true;
}

}  // namespace

auto EvaluateArm9FullSinkExperimentMatrix(bool omit_wirelength_unit) -> Arm9ExperimentMatrixRunResult
{
  const std::string_view env_name = ResolveArm9MatrixEnvName(omit_wirelength_unit);
  if (!ReadEnvFlag(env_name)) {
    return MakeSkipResult("Set " + std::string(env_name) + "=1 to run the ARM9 full-sink H-tree experiment matrix.");
  }

  const auto& setup_state = common_realtech::EnsureRealTechSetup();
  if (setup_state.mode != common_realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    return MakeSkipResult(setup_state.summary);
  }

  const auto selected_clock = SelectLargestRealClockLoads(std::numeric_limits<std::size_t>::max());
  if (!selected_clock.has_value()) {
    return MakeSkipResult("No DEF-derived clock net exposes at least two CTS sink pins.");
  }

  Arm9ExperimentMatrixRunResult matrix_result;
  matrix_result.selection = *selected_clock;
  if (!ValidateSelectedClock(matrix_result.selection, matrix_result.failure_messages)) {
    return matrix_result;
  }

  matrix_result.records.reserve(kArm9ExperimentIterations.size() * kArm9ExperimentSteps.size());
  for (const unsigned wirelength_iterations : kArm9ExperimentIterations) {
    for (const unsigned slew_cap_steps : kArm9ExperimentSteps) {
      const std::string scenario_name = MakeArm9CaseScenarioName(omit_wirelength_unit, wirelength_iterations, slew_cap_steps);

      realtech_fixture::RealTechCharFixture char_fixture;
      if (const auto prepare_error
          = char_fixture.prepare(scenario_name, std::nullopt, kHTreeSmokeMaxSlewNs, kHTreeSmokeMaxCapPf, omit_wirelength_unit);
          prepare_error.has_value()) {
        return MakeSkipResult(*prepare_error);
      }

      icts_test::runtime::CurrentRuntime().config.set_wirelength_iterations(wirelength_iterations);
      icts_test::runtime::CurrentRuntime().config.set_slew_steps(slew_cap_steps);
      icts_test::runtime::CurrentRuntime().config.set_cap_steps(slew_cap_steps);

      icts::Pin root_driver("htree_arm9_matrix_root_out", icts::PinType::kOut);
      icts::Net root_net("htree_arm9_matrix_root_net");
      ConnectRootNetForHTreeTest(root_net, root_driver, matrix_result.selection.loads);

      const auto runtime_start = std::chrono::steady_clock::now();
      const auto result = icts::htree::BuildWithDiagnostics(MakeExplicitHTreeInput(root_net), MakeExplicitHTreeConfig());
      const auto runtime_end = std::chrono::steady_clock::now();
      const double runtime_s = std::chrono::duration<double>(runtime_end - runtime_start).count();

      const Arm9ExperimentRecord record
          = MakeArm9ExperimentRecord(wirelength_iterations, slew_cap_steps, runtime_s, result, matrix_result.selection.loads.size());
      matrix_result.records.push_back(record);
      AppendArm9CaseFailures(wirelength_iterations, slew_cap_steps, omit_wirelength_unit, runtime_s, result, record,
                             matrix_result.failure_messages);
    }
  }

  const std::string_view scenario_name = ResolveArm9MatrixScenarioName(omit_wirelength_unit);
  matrix_result.report_written = realtech_fixture::WriteScenarioLog(
      std::string(scenario_name), "matrix_report.txt",
      FormatArm9ExperimentReport(scenario_name, matrix_result.selection.clock_name, matrix_result.selection.loads.size(),
                                 omit_wirelength_unit, matrix_result.records));
  return matrix_result;
}

}  // namespace icts_test
