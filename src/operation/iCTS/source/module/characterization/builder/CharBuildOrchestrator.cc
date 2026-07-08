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
 * @file CharBuildOrchestrator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Characterization sweep build driver and progress reporting.
 */

#include "characterization/builder/CharBuildOrchestrator.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "Log.hh"
#include "SegmentChar.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "characterization/pattern/CharPatternEnumerator.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"

namespace icts::char_builder::detail {
namespace {

auto formatFixed(double value, int precision = 4) -> std::string
{
  return logformat::FormatFixed(value, precision);
}

auto calcRatio(std::size_t numerator, std::size_t denominator) -> double
{
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

auto DetailStageReportOptions() -> StageReportOptions
{
  return StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail};
}

}  // namespace

auto CharBuildOrchestrator::build() -> void
{
  auto* reporter = _impl._reporter;
  std::optional<SchemaWriter::StageScope> build_stage;
  if (reporter != nullptr) {
    build_stage.emplace(reporter->beginStage("CharBuilder", "build", {}, DetailStageReportOptions()));
  }
  logformat::TableRows progress_rows;

  _impl._executed_sta_samples = 0U;
  _impl._skipped_sta_samples = 0U;
  _impl._output_slew_overflow_samples = 0U;
  _impl._driven_cap_overflow_samples = 0U;
  _impl._driven_cap_overflow_load_points = 0U;
  _impl._max_observed_output_slew_ns = 0.0;
  _impl._max_observed_output_slew_idx = 0U;
  _impl._max_observed_driven_cap_pf = 0.0;
  _impl._max_observed_driven_cap_idx = 0U;

  if (_impl._sorted_buffers.empty()) {
    LOG_WARNING << "CharBuilder: no usable buffers remain after Input/Config/liberty filtering, skip characterization build";
    if (build_stage.has_value()) {
      build_stage->skip({{"reason", "no_usable_buffers"}});
    }
    return;
  }
  if (_impl._wirelengths_um.empty()) {
    LOG_ERROR << "CharBuilder: no wirelengths to enumerate, aborting build";
    if (build_stage.has_value()) {
      build_stage->failed({{"reason", "no_wirelengths"}});
    }
    return;
  }
  if (_impl._slews_to_test.empty() || _impl._loads_to_test.empty()) {
    LOG_WARNING << "CharBuilder: characterization limits are unresolved"
                << " (max_slew=" << _impl._max_slew << " ns, max_cap=" << _impl._max_cap << " pF), skip characterization build";
    if (build_stage.has_value()) {
      build_stage->skip({{"reason", "unresolved_characterization_limits"}});
    }
    return;
  }

  for (std::size_t wirelength_index = 0; wirelength_index < _impl._wirelengths_um.size(); ++wirelength_index) {
    const unsigned length_idx = _impl._wirelength_indices.at(wirelength_index);
    const double wirelength_um = _impl._wirelengths_um.at(wirelength_index);
    const unsigned topology_slots = length_idx;
    const std::size_t estimated_patterns_per_wirelength = _impl.patternEnumerator().estimatePatternCountPerWirelength(wirelength_um);
    const std::size_t estimated_sta_samples_per_wirelength
        = estimated_patterns_per_wirelength * _impl._loads_to_test.size() * _impl._slews_to_test.size();
    const std::size_t char_count_before = _impl._segment_chars.size();
    const std::size_t pattern_count_before = _impl._buffering_patterns.size();
    BuildProgress build_progress;
    build_progress.wirelength_um = wirelength_um;
    build_progress.estimated_patterns = estimated_patterns_per_wirelength;
    build_progress.estimated_sta_samples = estimated_sta_samples_per_wirelength;
    if (build_stage.has_value()) {
      build_stage->markRunning("wirelength=" + formatFixed(wirelength_um) + " um",
                               {
                                   {"estimated_patterns", std::to_string(estimated_patterns_per_wirelength)},
                                   {"estimated_sta_samples", std::to_string(estimated_sta_samples_per_wirelength)},
                                   {"topology_slots", std::to_string(topology_slots)},
                               });
    }
    _impl.patternEnumerator().enumerateWirelength(length_idx, wirelength_um, build_progress);

    progress_rows.push_back({
        formatFixed(wirelength_um) + " um",
        std::to_string(topology_slots),
        std::to_string(_impl._segment_chars.size() - char_count_before),
        std::to_string(_impl._buffering_patterns.size() - pattern_count_before),
        std::to_string(build_progress.feasible_patterns),
        std::to_string(build_progress.skipped_patterns_infeasible),
        std::to_string(build_progress.executed_sta_samples),
        std::to_string(build_progress.skipped_sta_samples),
        std::to_string(build_progress.output_slew_overflow_samples),
        std::to_string(build_progress.driven_cap_overflow_samples),
        std::to_string(build_progress.driven_cap_overflow_load_points),
    });

    _impl._output_slew_overflow_samples += build_progress.output_slew_overflow_samples;
    _impl._executed_sta_samples += build_progress.executed_sta_samples;
    _impl._skipped_sta_samples += build_progress.skipped_sta_samples;
    _impl._driven_cap_overflow_samples += build_progress.driven_cap_overflow_samples;
    _impl._driven_cap_overflow_load_points += build_progress.driven_cap_overflow_load_points;
    _impl._max_observed_output_slew_ns = std::max(_impl._max_observed_output_slew_ns, build_progress.max_observed_output_slew_ns);
    _impl._max_observed_output_slew_idx = std::max(_impl._max_observed_output_slew_idx, build_progress.max_observed_output_slew_idx);
    _impl._max_observed_driven_cap_pf = std::max(_impl._max_observed_driven_cap_pf, build_progress.max_observed_driven_cap_pf);
    _impl._max_observed_driven_cap_idx = std::max(_impl._max_observed_driven_cap_idx, build_progress.max_observed_driven_cap_idx);

    LOG_INFO << "CharBuilder: [RUNNING] wirelength=" << formatFixed(wirelength_um)
             << " um, generated_chars=" << (_impl._segment_chars.size() - char_count_before)
             << ", generated_patterns=" << (_impl._buffering_patterns.size() - pattern_count_before)
             << ", feasible_patterns=" << build_progress.feasible_patterns
             << ", skipped_patterns=" << build_progress.skipped_patterns_infeasible
             << ", executed_sta_samples=" << build_progress.executed_sta_samples
             << ", skipped_sta_samples=" << build_progress.skipped_sta_samples
             << ", output_slew_overflow_samples=" << build_progress.output_slew_overflow_samples
             << ", driven_cap_overflow_samples=" << build_progress.driven_cap_overflow_samples
             << ", driven_cap_overflow_load_points=" << build_progress.driven_cap_overflow_load_points;
  }

  if (!progress_rows.empty() && reporter != nullptr) {
    reporter->emitTableTo(
        "CharBuilder Sweep Progress Detail",
        {"Wirelength", "Topology Slots", "Generated Chars", "Generated Patterns", "Feasible Patterns", "Skipped Patterns",
         "Executed STA Samples", "Skipped STA Samples", "Output Slew Overflow", "Driven Cap Overflow", "Driven Cap Overflow Load Points"},
        progress_rows, ReportSink::kDetail);
  }

  const double output_slew_overflow_ratio = calcRatio(_impl._output_slew_overflow_samples, _impl._executed_sta_samples);
  const double driven_cap_overflow_ratio = calcRatio(_impl._driven_cap_overflow_samples, _impl._executed_sta_samples);
  const KeyValueFields default_observed_fields = {
      {"segment_chars", std::to_string(_impl._segment_chars.size())},
      {"executed_sta_samples", std::to_string(_impl._executed_sta_samples)},
      {"skipped_sta_samples", std::to_string(_impl._skipped_sta_samples)},
      {"output_slew_overflow_samples", std::to_string(_impl._output_slew_overflow_samples)},
      {"output_slew_overflow_ratio", logformat::FormatPercent(output_slew_overflow_ratio, 2)},
      {"max_observed_output_slew", logformat::FormatWithUnit(_impl._max_observed_output_slew_ns, "ns")},
      {"driven_cap_overflow_samples", std::to_string(_impl._driven_cap_overflow_samples)},
      {"driven_cap_overflow_ratio", logformat::FormatPercent(driven_cap_overflow_ratio, 2)},
      {"driven_cap_overflow_load_points", std::to_string(_impl._driven_cap_overflow_load_points)},
      {"max_observed_driven_cap", logformat::FormatWithUnit(_impl._max_observed_driven_cap_pf, "pF")},
  };
  const KeyValueFields detail_observed_fields = {
      {"segment_chars", std::to_string(_impl._segment_chars.size())},
      {"executed_sta_samples", std::to_string(_impl._executed_sta_samples)},
      {"skipped_sta_samples", std::to_string(_impl._skipped_sta_samples)},
      {"output_slew_overflow_samples", std::to_string(_impl._output_slew_overflow_samples)},
      {"output_slew_overflow_ratio", logformat::FormatPercent(output_slew_overflow_ratio, 2)},
      {"max_observed_output_slew", logformat::FormatWithUnit(_impl._max_observed_output_slew_ns, "ns")},
      {"max_observed_output_slew_idx", std::to_string(_impl._max_observed_output_slew_idx)},
      {"slew_lattice_source", "CharBuilder Setup"},
      {"driven_cap_overflow_samples", std::to_string(_impl._driven_cap_overflow_samples)},
      {"driven_cap_overflow_ratio", logformat::FormatPercent(driven_cap_overflow_ratio, 2)},
      {"driven_cap_overflow_load_points", std::to_string(_impl._driven_cap_overflow_load_points)},
      {"max_observed_driven_cap", logformat::FormatWithUnit(_impl._max_observed_driven_cap_pf, "pF")},
      {"max_observed_driven_cap_idx", std::to_string(_impl._max_observed_driven_cap_idx)},
      {"cap_lattice_source", "CharBuilder Setup"},
  };
  if (reporter != nullptr) {
    reporter->emitSection("### Characterization Results");
    reporter->emitKeyValueTable("CharBuilder Results", default_observed_fields);
    reporter->emitKeyValueTableTo("CharBuilder Build Detail", detail_observed_fields, ReportSink::kDetail);
  }
  if (_impl._output_slew_overflow_samples > 0U) {
    if (reporter != nullptr) {
      EmitDiagnostic(*reporter, output_slew_overflow_ratio >= 0.10 ? DiagnosticLevel::kWarning : DiagnosticLevel::kInfo, "CharBuilder",
                     "output slew overflow occurred during characterization; samples were capped to the configured slew lattice.",
                     {
                         {"output_slew_overflow_samples", std::to_string(_impl._output_slew_overflow_samples)},
                         {"max_observed_output_slew", logformat::FormatWithUnit(_impl._max_observed_output_slew_ns, "ns")},
                         {"overflow_ratio_source", "CharBuilder Results"},
                         {"slew_lattice_source", "CharBuilder Setup"},
                     });
    }
  }

  if (build_stage.has_value()) {
    build_stage->finished({
        {"segment_chars", std::to_string(_impl._segment_chars.size())},
        {"patterns", std::to_string(_impl._buffering_patterns.size())},
    });
  }
}

}  // namespace icts::char_builder::detail
