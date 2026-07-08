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
 * @file CharSetupConfigurator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Characterization buffer discovery, limits, and sweep-grid initialization.
 */

#include "characterization/builder/CharSetupConfigurator.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "ClockRouteSegmentRc.hh"
#include "Log.hh"
#include "SegmentChar.hh"
#include "ValueLattice.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"

namespace icts::char_builder::detail {
namespace {

constexpr double kPercentFactor = 100.0;
constexpr unsigned kDefaultWirelengthIterations = 3U;

enum class ResolutionSource
{
  kRuntimeConfig,
  kCallerOverride,
  kLibertyPinLimit,
  kLibertyTableAxis,
  kAutoDerived,
  kUnresolved
};

struct ResolvedValue
{
  double value = 0.0;
  ResolutionSource source = ResolutionSource::kUnresolved;
  std::string detail;
};

auto toResolutionSourceName(ResolutionSource source) -> const char*
{
  switch (source) {
    case ResolutionSource::kRuntimeConfig:
      return "runtime_config";
    case ResolutionSource::kCallerOverride:
      return "caller_override";
    case ResolutionSource::kLibertyPinLimit:
      return "liberty_pin_limit";
    case ResolutionSource::kLibertyTableAxis:
      return "liberty_table_axis";
    case ResolutionSource::kAutoDerived:
      return "auto_derived";
    case ResolutionSource::kUnresolved:
      return "unresolved";
  }
  return "unresolved";
}

auto logInfoTable(SchemaWriter* reporter, const std::string& title, const std::vector<std::string>& headers,
                  const logformat::TableRows& rows) -> void
{
  if (reporter != nullptr) {
    EmitTable(*reporter, title, headers, rows);
  }
}

auto makeDenseWirelengthIndices(unsigned iterations) -> std::vector<unsigned>
{
  std::vector<unsigned> indices;
  indices.reserve(iterations);
  for (unsigned length_idx = 1U; length_idx <= iterations; ++length_idx) {
    indices.push_back(length_idx);
  }
  return indices;
}

auto normalizeWirelengthIndices(std::vector<unsigned> indices) -> std::vector<unsigned>
{
  std::erase(indices, 0U);
  std::ranges::sort(indices);
  const auto unique_tail = std::ranges::unique(indices);
  indices.erase(unique_tail.begin(), unique_tail.end());
  return indices;
}

auto clampWirelengthIndices(std::vector<unsigned> indices, unsigned max_length_idx) -> std::pair<std::vector<unsigned>, bool>
{
  indices = normalizeWirelengthIndices(std::move(indices));
  const auto old_size = indices.size();
  std::erase_if(indices, [&](unsigned length_idx) -> bool { return length_idx > max_length_idx; });
  const bool truncated = old_size != indices.size();
  return {std::move(indices), truncated};
}

auto makeWirelengths(double unit_um, const std::vector<unsigned>& indices) -> std::vector<double>
{
  std::vector<double> wirelengths_um;
  wirelengths_um.reserve(indices.size());
  for (const unsigned length_idx : indices) {
    wirelengths_um.push_back(static_cast<double>(length_idx) * unit_um);
  }
  return wirelengths_um;
}

auto resolveRoutingLayer(const ::icts::CharBuilder::Config& config) -> std::optional<int>
{
  if (config.routing_layer.has_value() && *config.routing_layer > 0) {
    return config.routing_layer;
  }
  LOG_FATAL << "CharBuilder: routing_layer must be explicitly provided.";
}

auto resolveWireWidth(const ::icts::CharBuilder::Config& config) -> std::optional<double>
{
  return config.wire_width_um.has_value() && *config.wire_width_um > 0.0 ? config.wire_width_um : std::nullopt;
}

auto resolveBufferDriveCap(const ::icts::CharacterizationBufferCell& buffer_cell) -> double
{
  double max_cap = buffer_cell.output_cap_limit_pf;
  if (max_cap <= 0.0) {
    max_cap = buffer_cell.output_cap_table_axis_max_pf;
  }
  return max_cap;
}

auto resolveClockRouteSegmentRc(const ::icts::CharBuilder::Input& input) -> ::icts::ClockRouteSegmentRc
{
  LOG_FATAL_IF(input.clock_route_segment_rc.capacitance_per_um_pf <= 0.0)
      << "CharBuilder: clock_route_segment_rc.capacitance_per_um_pf must be explicitly provided.";
  return input.clock_route_segment_rc;
}

auto resolveMaxSlew(const ::icts::CharBuilder::Input& input, const ::icts::CharBuilder::Config& config) -> ResolvedValue
{
  if (config.max_slew_ns.has_value() && *config.max_slew_ns > 0.0) {
    return ResolvedValue{
        .value = *config.max_slew_ns,
        .source = ResolutionSource::kRuntimeConfig,
        .detail = "CharBuilder config: max slew",
    };
  }

  double liberty_port_min_slew = std::numeric_limits<double>::infinity();
  bool found_port_limit = false;
  for (const auto& buffer_cell : input.characterization_buffer_cells) {
    if (buffer_cell.input_slew_limit_ns > 0.0) {
      liberty_port_min_slew = std::min(liberty_port_min_slew, buffer_cell.input_slew_limit_ns);
      found_port_limit = true;
    }
  }
  if (found_port_limit) {
    return ResolvedValue{
        .value = liberty_port_min_slew,
        .source = ResolutionSource::kLibertyPinLimit,
        .detail = "minimum liberty input pin slew limit",
    };
  }

  double liberty_table_min_slew = std::numeric_limits<double>::infinity();
  bool found_table_limit = false;
  for (const auto& buffer_cell : input.characterization_buffer_cells) {
    if (buffer_cell.input_slew_table_axis_max_ns > 0.0) {
      liberty_table_min_slew = std::min(liberty_table_min_slew, buffer_cell.input_slew_table_axis_max_ns);
      found_table_limit = true;
    }
  }
  if (found_table_limit) {
    return ResolvedValue{
        .value = liberty_table_min_slew,
        .source = ResolutionSource::kLibertyTableAxis,
        .detail = "minimum liberty input slew table-axis max",
    };
  }

  LOG_WARNING << "CharBuilder: failed to resolve max_slew from explicit config/liberty limits/liberty tables";
  return ResolvedValue{
      .value = 0.0,
      .source = ResolutionSource::kUnresolved,
      .detail = "missing explicit config/liberty slew limits",
  };
}

auto resolveMaxCap(const ::icts::CharBuilder::Input& input, const ::icts::CharBuilder::Config& config) -> ResolvedValue
{
  if (config.max_cap_pf.has_value() && *config.max_cap_pf > 0.0) {
    return ResolvedValue{
        .value = *config.max_cap_pf,
        .source = ResolutionSource::kRuntimeConfig,
        .detail = "CharBuilder config: max cap",
    };
  }

  double liberty_port_min_cap = std::numeric_limits<double>::infinity();
  bool found_port_limit = false;
  for (const auto& buffer_cell : input.characterization_buffer_cells) {
    if (buffer_cell.output_cap_limit_pf > 0.0) {
      liberty_port_min_cap = std::min(liberty_port_min_cap, buffer_cell.output_cap_limit_pf);
      found_port_limit = true;
    }
  }
  if (found_port_limit) {
    return ResolvedValue{
        .value = liberty_port_min_cap,
        .source = ResolutionSource::kLibertyPinLimit,
        .detail = "minimum liberty output pin cap limit",
    };
  }

  double liberty_table_min_cap = std::numeric_limits<double>::infinity();
  bool found_table_limit = false;
  for (const auto& buffer_cell : input.characterization_buffer_cells) {
    if (buffer_cell.output_cap_table_axis_max_pf > 0.0) {
      liberty_table_min_cap = std::min(liberty_table_min_cap, buffer_cell.output_cap_table_axis_max_pf);
      found_table_limit = true;
    }
  }
  if (found_table_limit) {
    return ResolvedValue{
        .value = liberty_table_min_cap,
        .source = ResolutionSource::kLibertyTableAxis,
        .detail = "minimum liberty output cap table-axis max",
    };
  }

  LOG_WARNING << "CharBuilder: failed to resolve max_cap from explicit config/liberty limits/liberty tables";
  return ResolvedValue{
      .value = 0.0,
      .source = ResolutionSource::kUnresolved,
      .detail = "missing explicit config/liberty cap limits",
  };
}

auto collectSortedBuffers(const ::icts::CharBuilder::Input& input, const ::icts::CharBuilder::Config& config)
    -> std::vector<::icts::CharacterizationBufferCell>
{
  std::vector<::icts::CharacterizationBufferCell> sorted_buffers;
  if (input.characterization_buffer_cells.empty()) {
    LOG_WARNING << "CharBuilder: no characterization buffer cells configured in explicit input";
    return sorted_buffers;
  }

  for (auto buffer_cell : input.characterization_buffer_cells) {
    const double max_cap = resolveBufferDriveCap(buffer_cell);
    if (max_cap <= 0.0) {
      LOG_WARNING << "CharBuilder: buffer " << buffer_cell.cell_master << " has invalid max_cap (" << max_cap << " pF), skipped";
      continue;
    }
    if (buffer_cell.input_cap_pf <= 0.0) {
      LOG_WARNING << "CharBuilder: buffer " << buffer_cell.cell_master << " has invalid input cap (" << buffer_cell.input_cap_pf
                  << " pF), skipped";
      continue;
    }
    if (buffer_cell.input_pin.empty() || buffer_cell.output_pin.empty()) {
      LOG_WARNING << "CharBuilder: buffer " << buffer_cell.cell_master << " has unresolved port names, skipped";
      continue;
    }

    buffer_cell.max_cap_pf = max_cap;
    sorted_buffers.push_back(std::move(buffer_cell));
  }

  std::ranges::sort(
      sorted_buffers,
      [](const ::icts::CharacterizationBufferCell& lhs_buffer_cell, const ::icts::CharacterizationBufferCell& rhs_buffer_cell) -> bool {
        return lhs_buffer_cell.max_cap_pf < rhs_buffer_cell.max_cap_pf;
      });

  const double buffer_redundancy_pct = config.char_buf_redundancy_pct.value_or(0.0);
  if (buffer_redundancy_pct > 0.0 && sorted_buffers.size() > 1U) {
    std::vector<::icts::CharacterizationBufferCell> filtered_buffers;
    filtered_buffers.push_back(sorted_buffers.front());
    for (std::size_t buffer_index = 1; buffer_index < sorted_buffers.size(); ++buffer_index) {
      const double prev_cap = filtered_buffers.back().max_cap_pf;
      const double curr_cap = sorted_buffers.at(buffer_index).max_cap_pf;
      if (prev_cap <= 0.0 || (curr_cap - prev_cap) / prev_cap >= buffer_redundancy_pct) {
        filtered_buffers.push_back(sorted_buffers.at(buffer_index));
      } else {
        LOG_INFO << "CharBuilder: removed near-neighbor buffer " << sorted_buffers.at(buffer_index).cell_master << " (max_cap=" << curr_cap
                 << " pF, gap=" << ((curr_cap - prev_cap) / prev_cap * kPercentFactor) << "%)";
      }
    }
    sorted_buffers = std::move(filtered_buffers);
  }

  return sorted_buffers;
}

auto resolveWirelengthUnitUm(SchemaWriter* reporter, const std::vector<::icts::CharacterizationBufferCell>& sorted_buffers) -> ResolvedValue
{
  double strongest_cap_pf = -1.0;
  double strongest_height_um = 0.0;
  std::string strongest_master;
  for (const auto& buffer_cell : sorted_buffers) {
    if (buffer_cell.cell_height_um <= 0.0) {
      LOG_WARNING << "CharBuilder: cannot derive wirelength_unit from " << buffer_cell.cell_master
                  << " because its physical height is unavailable";
      continue;
    }

    if (buffer_cell.max_cap_pf >= strongest_cap_pf) {
      strongest_cap_pf = buffer_cell.max_cap_pf;
      strongest_height_um = buffer_cell.cell_height_um;
      strongest_master = buffer_cell.cell_master;
    }
  }

  if (strongest_height_um <= 0.0) {
    LOG_WARNING << "CharBuilder: failed to resolve wirelength_unit from Input/Config or strongest buffer height";
    if (reporter != nullptr) {
      EmitDiagnostic(*reporter, DiagnosticLevel::kWarning, "CharBuilder",
                     "wirelength unit is absent in CharBuilder Input/Config and auto-derivation failed.",
                     {{"reason", "no_valid_strongest_buffer_height"}});
    }
    return ResolvedValue{
        .value = 0.0,
        .source = ResolutionSource::kUnresolved,
        .detail = "no valid strongest buffer height",
    };
  }

  const double auto_derived_unit_um = strongest_height_um * 10.0;
  if (reporter != nullptr) {
    EmitDiagnostic(*reporter, DiagnosticLevel::kWarning, "CharBuilder",
                   "wirelength unit is absent in CharBuilder Input/Config; explicit auto-derivation policy uses strongest-buffer height.",
                   {{"strongest_buffer", strongest_master},
                    {"buffer_height", logformat::FormatWithUnit(strongest_height_um, "um")},
                    {"derived_wirelength_unit", logformat::FormatWithUnit(auto_derived_unit_um, "um")}});
  }
  return ResolvedValue{
      .value = auto_derived_unit_um,
      .source = ResolutionSource::kAutoDerived,
      .detail = "strongest buffer " + strongest_master + " height=" + logformat::FormatWithUnit(strongest_height_um, "um") + " x10",
  };
}

auto resolveWirelengthUnitUm(SchemaWriter* reporter, const ::icts::CharBuilder::Config& config,
                             const std::vector<::icts::CharacterizationBufferCell>& sorted_buffers, bool caller_override) -> ResolvedValue
{
  if (config.wirelength_unit_um.has_value() && *config.wirelength_unit_um > 0.0) {
    return ResolvedValue{
        .value = *config.wirelength_unit_um,
        .source = caller_override ? ResolutionSource::kCallerOverride : ResolutionSource::kRuntimeConfig,
        .detail = caller_override ? "CharBuilder caller override: wirelength unit" : "runtime configuration",
    };
  }

  if (!config.allow_auto_wirelength_unit) {
    LOG_ERROR << "CharBuilder: wirelength_unit_um must be explicitly provided or allow_auto_wirelength_unit must be enabled.";
    if (reporter != nullptr) {
      EmitDiagnostic(*reporter, DiagnosticLevel::kError, "CharBuilder",
                     "wirelength unit is absent and auto-derivation is disabled by policy.", {{"reason", "missing_wirelength_unit"}});
    }
    return ResolvedValue{
        .value = 0.0,
        .source = ResolutionSource::kUnresolved,
        .detail = "missing explicit wirelength unit",
    };
  }

  return resolveWirelengthUnitUm(reporter, sorted_buffers);
}

auto DetailStageReportOptions() -> StageReportOptions
{
  return StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail};
}

}  // namespace

auto CharSetupConfigurator::init(const ::icts::CharBuilder::Input& input, const ::icts::CharBuilder::Config& config) -> void
{
  const ::icts::CharBuilder::Input& effective_input = input;
  const ::icts::CharBuilder::Config& effective_config = config;
  _impl._reporter = effective_input.reporter;
  auto* reporter = _impl._reporter;
  std::optional<SchemaWriter::StageScope> init_stage;
  if (reporter != nullptr) {
    init_stage.emplace(reporter->beginStage("CharBuilder", "initialization", {}, DetailStageReportOptions()));
  }

  _impl._segment_chars.clear();
  _impl._buffering_patterns.clear();
  _impl._wirelength_indices.clear();
  _impl._temp_inst_names.clear();
  _impl._temp_net_names.clear();
  _impl._source_inst_name.clear();
  _impl._source_in_pin.clear();
  _impl._sink_inst_name.clear();
  _impl._source_out_pin.clear();
  _impl._sink_in_pin.clear();
  _impl._char_clock_name.clear();
  _impl._next_pattern_id = 0U;
  _impl._char_circuit_id = 0U;
  _impl._executed_sta_samples = 0U;
  _impl._skipped_sta_samples = 0U;
  _impl._output_slew_overflow_samples = 0U;
  _impl._driven_cap_overflow_samples = 0U;
  _impl._driven_cap_overflow_load_points = 0U;
  _impl._max_observed_output_slew_ns = 0.0;
  _impl._max_observed_output_slew_idx = 0U;
  _impl._max_observed_driven_cap_pf = 0.0;
  _impl._max_observed_driven_cap_idx = 0U;
  _impl._fast_sta = effective_input.fast_sta;
  _impl._wrapper = effective_input.wrapper;
  _impl._dbu_per_um = effective_input.dbu_per_um;
  _impl._root_input_slew_ns = std::max(0.0, effective_input.root_input_slew_ns);
  LOG_FATAL_IF(_impl._wrapper == nullptr) << "CharBuilder: Wrapper dependency must be explicitly provided.";
  LOG_FATAL_IF(_impl._fast_sta == nullptr) << "CharBuilder: FastSTA dependency must be explicitly provided.";

  _impl._sorted_buffers = collectSortedBuffers(effective_input, effective_config);
  const ResolvedValue max_slew_resolution = resolveMaxSlew(effective_input, effective_config);
  const ResolvedValue max_cap_resolution = resolveMaxCap(effective_input, effective_config);
  const ResolvedValue wirelength_unit_resolution
      = resolveWirelengthUnitUm(reporter, effective_config, _impl._sorted_buffers, config.wirelength_unit_um.has_value());
  const auto routing_layer_resolution = resolveRoutingLayer(effective_config);
  const auto clock_route_segment_rc = resolveClockRouteSegmentRc(effective_input);
  _impl._max_slew = max_slew_resolution.value;
  _impl._max_cap = max_cap_resolution.value;
  _impl._length_unit_um = wirelength_unit_resolution.value;
  _impl._wirelength_unit_source = toResolutionSourceName(wirelength_unit_resolution.source);
  _impl._wirelength_unit_detail = wirelength_unit_resolution.detail;
  _impl._wirelength_iterations = std::max(1U, effective_config.wirelength_iterations.value_or(kDefaultWirelengthIterations));
  _impl._slew_steps = effective_config.slew_steps.value_or(15U);
  _impl._cap_steps = effective_config.cap_steps.value_or(15U);
  if (_impl._max_slew <= 0.0 || _impl._max_cap <= 0.0 || _impl._length_unit_um <= 0.0 || !routing_layer_resolution.has_value()) {
    std::string reason = "max_cap_unresolved";
    if (_impl._length_unit_um <= 0.0) {
      reason = "wirelength_unit_unresolved";
    } else if (!routing_layer_resolution.has_value()) {
      reason = "routing_layer_unresolved";
    } else if (_impl._max_slew <= 0.0) {
      reason = "max_slew_unresolved";
    }
    if (init_stage.has_value()) {
      init_stage->failed({{"reason", reason}});
    }
    return;
  }
  bool wirelength_indices_truncated = false;
  if (effective_config.wirelength_indices.has_value()) {
    auto [clamped_indices, truncated] = clampWirelengthIndices(*effective_config.wirelength_indices, _impl._wirelength_iterations);
    _impl._wirelength_indices = std::move(clamped_indices);
    wirelength_indices_truncated = truncated;
  }
  if (_impl._wirelength_indices.empty()) {
    _impl._wirelength_indices = makeDenseWirelengthIndices(_impl._wirelength_iterations);
  }
  _impl._wirelengths_um = makeWirelengths(_impl._length_unit_um, _impl._wirelength_indices);
  _impl._max_length = _impl._wirelengths_um.empty() ? 0.0 : _impl._wirelengths_um.back();
  const ::icts::UniformValueLattice slew_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_slew, _impl._slew_steps);
  const ::icts::UniformValueLattice cap_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_cap, _impl._cap_steps);
  _impl._slews_to_test = slew_lattice.sampleValues();
  _impl._loads_to_test = cap_lattice.sampleValues();
  _impl._routing_layer = *routing_layer_resolution;
  _impl._wire_width_um = resolveWireWidth(effective_config);
  _impl._clock_route_segment_rc = clock_route_segment_rc;
  _impl._sink_input_cap_pf = _impl._sorted_buffers.empty() ? 0.0 : _impl._sorted_buffers.front().input_cap_pf;

  if (wirelength_indices_truncated) {
    if (reporter != nullptr) {
      EmitDiagnostic(*reporter, DiagnosticLevel::kWarning, "CharBuilder",
                     "wirelength_indices exceeded wirelength_iterations; clamp direct characterization to the configured max iter.",
                     {
                         {"wirelength_iterations", std::to_string(_impl._wirelength_iterations)},
                         {"wirelength_points", std::to_string(_impl._wirelength_indices.size())},
                     });
    }
  }

  if (reporter != nullptr) {
    reporter->emitSection("### Characterization Setup");
  }
  logformat::TableRows default_setup_rows;
  if (!effective_config.wirelength_indices.has_value()) {
    default_setup_rows.push_back({"resolved_wirelength_unit", logformat::FormatWithUnit(_impl._length_unit_um, "um"),
                                  toResolutionSourceName(wirelength_unit_resolution.source)});
  }
  default_setup_rows.push_back(
      {"wirelength_points", std::to_string(_impl._wirelengths_um.size()),
       effective_config.wirelength_indices.has_value() ? "HTree Characterization Grid Plan" : wirelength_unit_resolution.detail});
  default_setup_rows.push_back({"routing_rc", "Runtime Routing / Wire RC", "CharBuilderInput"});
  default_setup_rows.push_back({"buffer_count", std::to_string(_impl._sorted_buffers.size()), "resolved_buffers"});
  const logformat::TableRows detail_setup_rows = {
      {"max_slew", logformat::FormatWithUnit(_impl._max_slew, "ns"), toResolutionSourceName(max_slew_resolution.source)},
      {"max_cap", logformat::FormatWithUnit(_impl._max_cap, "pF"), toResolutionSourceName(max_cap_resolution.source)},
      {"resolved_wirelength_unit", logformat::FormatWithUnit(_impl._length_unit_um, "um"),
       effective_config.wirelength_indices.has_value() ? "HTree Characterization Grid Plan"
                                                       : toResolutionSourceName(wirelength_unit_resolution.source)},
      {"wirelength_points", std::to_string(_impl._wirelengths_um.size()),
       effective_config.wirelength_indices.has_value() ? "HTree Characterization Grid Plan" : wirelength_unit_resolution.detail},
      {"slew_steps", std::to_string(_impl._slew_steps), "Runtime Configuration"},
      {"cap_steps", std::to_string(_impl._cap_steps), "Runtime Configuration"},
      {"routing_rc", "Runtime Routing / Wire RC", "CharBuilderInput"},
      {"buffer_count", std::to_string(_impl._sorted_buffers.size()), "resolved_buffers"},
  };
  logInfoTable(reporter, "CharBuilder Setup", {"Parameter", "Value", "Source"}, default_setup_rows);
  if (reporter != nullptr) {
    reporter->emitTableTo("CharBuilder Setup Detail", {"Parameter", "Value", "Source"}, detail_setup_rows, ReportSink::kDetail);
  }

  logformat::TableRows buffer_rows;
  buffer_rows.reserve(_impl._sorted_buffers.size());
  for (std::size_t buffer_index = 0; buffer_index < _impl._sorted_buffers.size(); ++buffer_index) {
    buffer_rows.push_back({
        std::to_string(buffer_index),
        _impl._sorted_buffers.at(buffer_index).cell_master,
        logformat::FormatWithUnit(_impl._sorted_buffers.at(buffer_index).max_cap_pf, "pF"),
        logformat::FormatWithUnit(_impl._sorted_buffers.at(buffer_index).input_cap_pf, "pF"),
    });
  }
  if (!buffer_rows.empty()) {
    logInfoTable(reporter, "CharBuilder Sorted Buffers", {"Index", "Cell Master", "Max Cap", "Input Cap"}, buffer_rows);
  }
  if (init_stage.has_value()) {
    init_stage->finished({
        {"buffers", std::to_string(_impl._sorted_buffers.size())},
        {"wirelengths", std::to_string(_impl._wirelengths_um.size())},
        {"slews", std::to_string(_impl._slews_to_test.size())},
        {"loads", std::to_string(_impl._loads_to_test.size())},
    });
  }
}

}  // namespace icts::char_builder::detail
