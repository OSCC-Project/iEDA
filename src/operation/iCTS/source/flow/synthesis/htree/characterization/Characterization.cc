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
 * @file Characterization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief H-tree characterization grid setup and CharBuilder result capture.
 */

#include "synthesis/htree/characterization/Characterization.hh"

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "LogFormat.hh"
#include "characterization/Characterization.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/htree/characterization/wirelength/WirelengthGrid.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"

namespace icts {
class Tree;
}  // namespace icts

namespace icts::htree {

namespace {

auto AppendPositiveLengths(std::vector<double>& target, const std::vector<double>& values) -> void
{
  for (const double value : values) {
    if (value > 0.0) {
      target.push_back(value);
    }
  }
}

auto AppendUniqueLengthIndex(std::vector<unsigned>& target, unsigned length_idx) -> void
{
  if (length_idx == 0U) {
    return;
  }
  for (const unsigned existing_idx : target) {
    if (existing_idx == length_idx) {
      return;
    }
  }
  target.push_back(length_idx);
}

auto FormatLogValue(const std::string& value) -> std::string
{
  return value.empty() ? "n/a" : value;
}

auto MakeContextFields(const HTree::LogContext& context, const std::string& object_name_prefix) -> KeyValueFields
{
  return {
      {"clock_name", FormatLogValue(context.clock_name)},
      {"clock_net_name", FormatLogValue(context.clock_net_name)},
      {"sink_domain", FormatLogValue(context.sink_domain)},
      {"stage", FormatLogValue(context.stage)},
      {"object_name_prefix", FormatLogValue(object_name_prefix.empty() ? context.object_name_prefix : object_name_prefix)},
  };
}

}  // namespace

auto RunCharacterizationFlow(const Tree& topology, int32_t dbu_per_um, const CharBuilder::Input& base_char_input,
                             const CharBuilder::Config& base_char_config, htree::DiagnosticBuild& result,
                             CharacterizationLibrary& char_library, const HTree::Input& input, const HTree::Config& config)
    -> CharacterizationSummary
{
  auto requested_lengths_um = CollectRequestedLevelLengthsUm(topology, dbu_per_um);
  AppendPositiveLengths(requested_lengths_um, input.additional_characterization_lengths_um);
  LOG_FATAL_IF(input.reporter == nullptr) << "HTree characterization requires an explicit reporter.";
  auto& reporter = *input.reporter;
  const auto char_grid_plan = ResolveCharacterizationGridPlan(base_char_config, requested_lengths_um);
  std::string grid_source = ToCharGridSourceName(CharGridSource::kNone);
  if (char_grid_plan.adapted) {
    grid_source = ToCharGridSourceName(char_grid_plan.source);
  } else if (char_grid_plan.configured_wirelength_unit_um > 0.0) {
    grid_source = ToCharGridSourceName(CharGridSource::kRuntimeConfig);
  }

  std::vector<std::string> decision_flag_values;
  if (char_grid_plan.configured_wirelength_missing && char_grid_plan.configured_grid_collapsed) {
    decision_flag_values.emplace_back("missing_config");
    decision_flag_values.emplace_back("collapsed_bins");
  } else if (char_grid_plan.configured_wirelength_missing) {
    decision_flag_values.emplace_back("missing_config");
  } else if (char_grid_plan.configured_grid_collapsed) {
    decision_flag_values.emplace_back("collapsed_bins");
  }
  if (char_grid_plan.adapted && char_grid_plan.wirelength_iterations < char_grid_plan.required_covering_iterations) {
    decision_flag_values.emplace_back("direct_bins_capped");
  }
  const std::string decision_flags = decision_flag_values.empty() ? std::string{"none"} : logformat::JoinStrings(decision_flag_values, "+");
  logformat::TableRows grid_plan_rows = {
      {"source", grid_source, char_grid_plan.adapted ? "auto-derived from topology level lengths" : "use runtime-configured grid"},
      {"requested_level_lengths", std::to_string(char_grid_plan.requested_level_lengths),
       "average parent-child segment length per topology level plus caller-supplied source-to-root lengths"},
      {"required_covering_iterations", std::to_string(char_grid_plan.required_covering_iterations),
       char_grid_plan.adapted ? "cover all topology level lengths under the resolved CharBuilder unit" : "0 (disabled)"},
      {"direct_characterization_bins", std::to_string(char_grid_plan.wirelength_iterations),
       char_grid_plan.adapted ? "direct-char bins after runtime cap" : "0 (disabled)"},
      {"distinct_level_bins", std::to_string(char_grid_plan.unique_level_bins), "aligned-length bins under resolved setup"},
      {"decision_flags", decision_flags, "auto-derivation/adaptation trigger flags"},
  };
  if (char_grid_plan.adapted) {
    grid_plan_rows.insert(grid_plan_rows.begin() + 1,
                          {"resolved_wirelength_unit", logformat::FormatWithUnit(char_grid_plan.wirelength_unit_um, "um"),
                           "effective unit for the adapted characterization grid"});
  }
  reporter.emitSection("### H-Tree Characterization");
  reporter.emitKeyValueTableTo("HTree Build Scope", MakeContextFields(input.log_context, input.object_name_prefix), ReportSink::kBoth);
  EmitTable(reporter, "HTree Characterization Grid Plan", {"Item", "Value", "Detail"}, grid_plan_rows);

  auto char_config = base_char_config;
  if (char_grid_plan.adapted) {
    char_config.wirelength_unit_um = char_grid_plan.wirelength_unit_um;
    char_config.wirelength_iterations = char_grid_plan.wirelength_iterations;
    auto direct_length_indices = ResolveDirectCharacterizationLengthIndices(requested_lengths_um, char_grid_plan);
    if (config.enable_analytical_solver) {
      AppendUniqueLengthIndex(direct_length_indices, 1U);
    }
    if (!direct_length_indices.empty()) {
      char_config.wirelength_indices = std::move(direct_length_indices);
    }
  }
  const auto ensure_result = char_library.ensure(base_char_input, char_config);
  if (!ensure_result.success) {
    return CharacterizationSummary{
        .success = false,
        .failure_reason = ensure_result.failure_reason.empty() ? "characterization_library_failed" : ensure_result.failure_reason,
        .length_step_um = 0.0};
  }

  const auto& char_builder = char_library.getCharBuilder();

  const double length_step_um = char_builder.get_wirelength_unit_um();
  if (length_step_um <= 0.0 || char_builder.get_segment_chars().empty()) {
    LOG_WARNING << "HTree: characterization did not produce usable segment chars.";
    return CharacterizationSummary{.success = false, .failure_reason = "no_usable_segment_chars", .length_step_um = length_step_um};
  }

  result.diagnostics.char_wirelength_unit_um = length_step_um;
  result.diagnostics.char_wirelength_iterations = char_builder.get_wirelength_iterations();
  result.diagnostics.char_unique_level_bins
      = char_grid_plan.adapted ? char_grid_plan.unique_level_bins
                               : CountUniqueAlignedLengthBins(CollectRequestedLevelLengthsUm(topology, dbu_per_um), length_step_um);
  result.diagnostics.char_grid_adapted = char_grid_plan.adapted;
  result.diagnostics.char_max_slew_ns = char_builder.get_max_slew();
  result.diagnostics.char_max_cap_pf = char_builder.get_max_cap();
  result.diagnostics.char_slew_steps = char_builder.get_slew_steps();
  result.diagnostics.char_cap_steps = char_builder.get_cap_steps();
  logformat::TableRows char_summary_rows = {
      {"characterization_setup_source", "CharBuilder Setup", "resolved limits, wirelength lattice, buffers, and routing source"},
      {"characterization_results_source", "CharBuilder Results",
       "sample counts and overflow ratios are logged by the characterization owner"},
      {"wirelength_setup_source", char_grid_plan.adapted ? "HTree Characterization Grid Plan" : "runtime_setup",
       char_grid_plan.adapted ? "effective wirelength lattice came from topology-grid adaptation"
                              : "runtime config supplied the characterization lattice"},
      {"grid_plan_source", grid_source, char_grid_plan.adapted ? "HTree adapted characterization grid" : "no HTree override"},
      {"characterization_library", ensure_result.reused ? "reused" : "built", "shared characterization cache state"},
  };
  reporter.emitTableTo("HTree Characterization Detail", {"Item", "Value", "Detail"}, char_summary_rows, ReportSink::kDetail);

  return CharacterizationSummary{.success = true, .failure_reason = {}, .length_step_um = length_step_um};
}

}  // namespace icts::htree
