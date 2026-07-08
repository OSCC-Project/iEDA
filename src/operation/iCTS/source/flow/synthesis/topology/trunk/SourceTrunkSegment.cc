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
 * @file SourceTrunkSegment.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Source-to-sink segment synthesis using reusable characterization frontiers.
 */

#include "synthesis/topology/trunk/SourceTrunkSegment.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "Inst.hh"
#include "Log.hh"
#include "LogFormat.hh"
#include "Net.hh"
#include "PatternId.hh"
#include "Pin.hh"
#include "Point.hh"
#include "SegmentChar.hh"
#include "ValueLattice.hh"
#include "characterization/Characterization.hh"
#include "geometry/Geometry.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/characterization/Characterization.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/htree/characterization/wirelength/WirelengthGrid.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/embedding/BufferPortTable.hh"
#include "synthesis/htree/embedding/Embedding.hh"
#include "synthesis/htree/segment_pruning/SegmentFrontierCatalog.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"
#include "synthesis/htree/segment_pruning/SegmentPruning.hh"

namespace icts {
namespace {

auto FormatLogValue(const std::string& value) -> std::string
{
  return value.empty() ? "n/a" : value;
}

auto DetailStageReportOptions() -> StageReportOptions
{
  return StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail};
}

auto MakeObjectName(const std::string& prefix, const std::string& suffix) -> std::string
{
  return prefix.empty() ? "cts_" + suffix : prefix + "_" + suffix;
}

auto ConnectNet(Net& net, Pin* driver, const std::vector<Pin*>& loads) -> void
{
  auto* old_driver = net.get_driver();
  if (old_driver != nullptr && old_driver != driver && old_driver->get_net() == &net) {
    old_driver->set_net(nullptr);
  }

  for (auto* old_load : net.get_loads()) {
    if (old_load != nullptr && old_load->get_net() == &net) {
      old_load->set_net(nullptr);
    }
  }

  net.set_driver(driver);
  if (driver != nullptr) {
    driver->set_net(&net);
  }

  net.set_loads({});
  for (auto* load : loads) {
    if (load == nullptr) {
      continue;
    }
    net.add_load(load);
    load->set_net(&net);
  }
}

auto ConnectOwnedNet(Net& net, Pin* driver, const std::vector<Pin*>& loads) -> void
{
  net.set_driver(driver);
  if (driver != nullptr) {
    driver->set_net(&net);
  }

  net.set_loads({});
  for (auto* load : loads) {
    if (load == nullptr) {
      continue;
    }
    net.add_load(load);
    load->set_net(&net);
  }
}

auto CreateBufferInstance(SourceTrunkSegment::Build& result, const std::string& inst_name, const std::string& cell_master,
                          const Point<int>& location, const std::string& input_pin_name, const std::string& output_pin_name)
    -> std::pair<Pin*, Pin*>
{
  auto inst = std::make_unique<Inst>(inst_name, cell_master, InstType::kBuffer, location);
  auto* inst_ptr = inst.get();

  auto input_pin = std::make_unique<Pin>(input_pin_name, PinType::kIn, location, inst_ptr, nullptr, false);
  auto* input_pin_ptr = input_pin.get();
  result.output.inserted_pins.push_back(std::move(input_pin));

  auto output_pin = std::make_unique<Pin>(output_pin_name, PinType::kOut, location, inst_ptr, nullptr, false);
  auto* output_pin_ptr = output_pin.get();
  result.output.inserted_pins.push_back(std::move(output_pin));

  inst_ptr->add_pin(input_pin_ptr);
  inst_ptr->insertDriverPin(output_pin_ptr);
  result.output.inserted_insts.push_back(std::move(inst));

  return {input_pin_ptr, output_pin_ptr};
}

auto RecordInsertedInstLevel(SourceTrunkSegment::Build& result, Inst* inst, int topology_level, std::size_t index_in_level) -> void
{
  if (inst == nullptr) {
    return;
  }
  result.output.inserted_inst_levels.push_back(HTree::InsertedInstLevel{
      .inst = inst,
      .topology_level = topology_level,
      .index_in_level = index_in_level,
  });
}

auto RecordInsertedNetLevel(SourceTrunkSegment::Build& result, Net* net, int topology_level, std::size_t index_in_level) -> void
{
  if (net == nullptr) {
    return;
  }
  result.output.inserted_net_levels.push_back(HTree::InsertedNetLevel{
      .net = net,
      .topology_level = topology_level,
      .index_in_level = index_in_level,
  });
}

auto CreateNet(SourceTrunkSegment::Build& result, const std::string& net_name, Pin* driver, const std::vector<Pin*>& loads,
               int topology_level, std::size_t index_in_level) -> Net*
{
  auto net = std::make_unique<Net>(net_name);
  auto* net_ptr = net.get();
  ConnectOwnedNet(*net_ptr, driver, loads);
  RecordInsertedNetLevel(result, net_ptr, topology_level, index_in_level);
  result.output.inserted_nets.push_back(std::move(net));
  return net_ptr;
}

auto DelayPowerDominates(const SegmentChar& lhs, const SegmentChar& rhs) -> bool
{
  const bool not_worse = lhs.get_delay() <= rhs.get_delay() && lhs.get_power() <= rhs.get_power();
  const bool strictly_better = lhs.get_delay() < rhs.get_delay() || lhs.get_power() < rhs.get_power();
  return not_worse && strictly_better;
}

auto PreferSegmentEntry(const SegmentChar& lhs, const SegmentChar& rhs) -> bool
{
  if (lhs.get_power() != rhs.get_power()) {
    return lhs.get_power() < rhs.get_power();
  }
  if (lhs.get_delay() != rhs.get_delay()) {
    return lhs.get_delay() < rhs.get_delay();
  }
  if (lhs.get_driven_cap_idx() != rhs.get_driven_cap_idx()) {
    return lhs.get_driven_cap_idx() < rhs.get_driven_cap_idx();
  }
  if (lhs.get_output_slew_idx() != rhs.get_output_slew_idx()) {
    return lhs.get_output_slew_idx() < rhs.get_output_slew_idx();
  }
  if (lhs.get_load_cap_idx() != rhs.get_load_cap_idx()) {
    return lhs.get_load_cap_idx() < rhs.get_load_cap_idx();
  }
  if (lhs.get_input_slew_idx() != rhs.get_input_slew_idx()) {
    return lhs.get_input_slew_idx() > rhs.get_input_slew_idx();
  }
  return lhs.get_pattern_id().pack() < rhs.get_pattern_id().pack();
}

auto BuildDelayPowerParetoFront(const std::vector<SegmentChar>& entries) -> std::vector<SegmentChar>
{
  std::vector<SegmentChar> pareto_front;
  pareto_front.reserve(entries.size());
  for (std::size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
    bool dominated = false;
    for (std::size_t other_index = 0; other_index < entries.size(); ++other_index) {
      if (entry_index == other_index) {
        continue;
      }
      if (DelayPowerDominates(entries.at(other_index), entries.at(entry_index))) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      pareto_front.push_back(entries.at(entry_index));
    }
  }

  std::ranges::sort(pareto_front, PreferSegmentEntry);
  return pareto_front;
}

auto SelectBestSegmentEntry(const std::vector<SegmentChar>& entries) -> std::optional<SegmentChar>
{
  if (entries.empty()) {
    return std::nullopt;
  }
  auto pareto_front = BuildDelayPowerParetoFront(entries);
  if (pareto_front.empty()) {
    return std::nullopt;
  }
  const std::size_t median_index = (pareto_front.size() - 1U) / 2U;
  return pareto_front.at(median_index);
}

auto FilterSegmentEntries(const std::vector<SegmentChar>& entries, unsigned required_load_cap_idx, unsigned source_drive_cap_idx,
                          const std::optional<unsigned>& min_input_slew_idx) -> std::vector<SegmentChar>
{
  std::vector<SegmentChar> filtered_entries;
  filtered_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    if (entry.get_load_cap_idx() < required_load_cap_idx) {
      continue;
    }
    if (entry.get_driven_cap_idx() > source_drive_cap_idx) {
      continue;
    }
    if (min_input_slew_idx.has_value() && entry.get_input_slew_idx() < *min_input_slew_idx) {
      continue;
    }
    filtered_entries.push_back(entry);
  }
  return filtered_entries;
}

auto BuildSourceTrunkSegmentObjects(SourceTrunkSegment::Build& result, Net& source_net, Pin* source, Pin* sink,
                                    const BufferingPattern& pattern, const SourceTrunkSegment::Input& input) -> bool
{
  const auto& cell_masters = pattern.get_cell_masters();
  const auto& positions = pattern.get_buffer_positions();
  const std::size_t buffer_count = std::min(cell_masters.size(), positions.size());
  if (buffer_count == 0U) {
    ConnectNet(source_net, source, {sink});
    return true;
  }

  htree::BufferPortTable port_table(*input.wrapper);
  std::vector<std::pair<Pin*, Pin*>> segment_buffers;
  segment_buffers.reserve(buffer_count);
  for (std::size_t buffer_index = 0; buffer_index < buffer_count; ++buffer_index) {
    const auto* ports = port_table.get(cell_masters.at(buffer_index));
    if (ports == nullptr) {
      LOG_WARNING << "SourceTrunkSegment: unresolved ports for source-to-root buffer master " << cell_masters.at(buffer_index) << ".";
      result.summary.failure_reason = "unresolved_buffer_ports";
      return false;
    }

    const auto location = htree::InterpolateManhattanPoint(source->get_location(), sink->get_location(), positions.at(buffer_index));
    auto created_buffer
        = CreateBufferInstance(result, MakeObjectName(input.object_name_prefix, "top_segment_buf_" + std::to_string(buffer_index)),
                               cell_masters.at(buffer_index), location, ports->first, ports->second);
    RecordInsertedInstLevel(result, created_buffer.first == nullptr ? nullptr : created_buffer.first->get_inst(),
                            static_cast<int>(buffer_index), buffer_index);
    segment_buffers.push_back(created_buffer);
  }

  ConnectNet(source_net, source, {segment_buffers.front().first});
  for (std::size_t buffer_index = 0; buffer_index + 1U < segment_buffers.size(); ++buffer_index) {
    CreateNet(result, MakeObjectName(input.object_name_prefix, "top_segment_net_" + std::to_string(buffer_index)),
              segment_buffers.at(buffer_index).second, {segment_buffers.at(buffer_index + 1U).first}, static_cast<int>(buffer_index),
              buffer_index);
  }
  CreateNet(result, MakeObjectName(input.object_name_prefix, "top_segment_net_" + std::to_string(segment_buffers.size() - 1U)),
            segment_buffers.back().second, {sink}, static_cast<int>(segment_buffers.size() - 1U), segment_buffers.size() - 1U);
  return true;
}

auto EmitSegmentSummary(const SourceTrunkSegment::Build& result, const SourceTrunkSegment::Input& input,
                        const SourceTrunkSegment::Config& config) -> void
{
  LOG_FATAL_IF(input.reporter == nullptr) << "SourceTrunkSegment summary requires an explicit reporter.";
  auto& reporter = *input.reporter;
  std::vector<std::string> default_headers = {"Clock",
                                              "Net",
                                              "Stage",
                                              "Status",
                                              "Length",
                                              "Required Load Cap",
                                              "Source Drive",
                                              "Strict Candidates",
                                              "Relaxed Candidates",
                                              "Inserted Insts",
                                              "Inserted Nets"};
  logformat::TableRows default_rows = {{
      FormatLogValue(input.log_context.clock_name),
      FormatLogValue(input.log_context.clock_net_name),
      FormatLogValue(input.log_context.stage),
      result.summary.success ? "finished" : "failed",
      logformat::FormatWithUnit(result.summary.length_um, "um"),
      logformat::FormatWithUnit(input.required_load_cap_pf, "pF"),
      logformat::FormatWithUnit(input.source_drive_cap_pf, "pF"),
      std::to_string(result.summary.strict_candidate_count),
      std::to_string(result.summary.relaxed_candidate_count),
      std::to_string(result.output.inserted_insts.size()),
      std::to_string(result.output.inserted_nets.size()),
  }};
  if (result.summary.used_boundary_relaxation) {
    default_headers.emplace_back("Relaxation Reason");
    default_rows.front().emplace_back(result.summary.boundary_relaxation_reason.empty() ? "unknown"
                                                                                        : result.summary.boundary_relaxation_reason);
  }
  if (!result.summary.success) {
    default_headers.emplace_back("Failure");
    default_rows.front().emplace_back(result.summary.failure_reason.empty() ? "source_trunk_segment_failed"
                                                                            : result.summary.failure_reason);
  }
  reporter.emitTableTo("Source Trunk Summary", default_headers, default_rows, ReportSink::kDefault);

  KeyValueFields detail_fields = {
      {"clock_name", FormatLogValue(input.log_context.clock_name)},
      {"clock_net_name", FormatLogValue(input.log_context.clock_net_name)},
      {"sink_domain", FormatLogValue(input.log_context.sink_domain)},
      {"stage", FormatLogValue(input.log_context.stage)},
      {"object_name_prefix", FormatLogValue(input.object_name_prefix)},
      {"status", result.summary.success ? "finished" : "failed"},
      {"length", logformat::FormatWithUnit(result.summary.length_um, "um")},
      {"required_load_cap", logformat::FormatWithUnit(input.required_load_cap_pf, "pF")},
      {"source_drive_cap", logformat::FormatWithUnit(input.source_drive_cap_pf, "pF")},
      {"min_input_slew", config.min_input_slew_ns.has_value() ? logformat::FormatWithUnit(*config.min_input_slew_ns, "ns") : "none"},
      {"length_idx", std::to_string(result.summary.length_idx)},
      {"required_load_cap_idx", std::to_string(result.summary.required_load_cap_idx)},
      {"source_drive_cap_idx", std::to_string(result.summary.source_drive_cap_idx)},
      {"min_input_slew_idx", result.summary.min_input_slew_idx.has_value() ? std::to_string(*result.summary.min_input_slew_idx) : "none"},
      {"strict_candidate_count", std::to_string(result.summary.strict_candidate_count)},
      {"relaxed_candidate_count", std::to_string(result.summary.relaxed_candidate_count)},
      {"used_boundary_relaxation", logformat::FormatBool(result.summary.used_boundary_relaxation)},
      {"boundary_relaxation_reason",
       result.summary.boundary_relaxation_reason.empty() ? "none" : result.summary.boundary_relaxation_reason},
      {"segment_inserted_insts", std::to_string(result.output.inserted_insts.size())},
      {"segment_inserted_nets", std::to_string(result.output.inserted_nets.size())},
  };
  if (!result.summary.failure_reason.empty()) {
    detail_fields.emplace_back("failure_reason", result.summary.failure_reason);
  }
  reporter.emitKeyValueTableTo("SourceTrunkSegment Build Detail", detail_fields, ReportSink::kDetail);
}

auto ConfigureCharConfig(const CharBuilder::Config& base_config, const std::vector<double>& requested_lengths_um) -> CharBuilder::Config
{
  auto char_config = base_config;
  const auto char_grid_plan = htree::ResolveCharacterizationGridPlan(base_config, requested_lengths_um);
  if (!char_grid_plan.adapted) {
    return char_config;
  }

  char_config.wirelength_unit_um = char_grid_plan.wirelength_unit_um;
  char_config.wirelength_iterations = char_grid_plan.wirelength_iterations;
  auto direct_indices = htree::ResolveDirectCharacterizationLengthIndices(requested_lengths_um, char_grid_plan);
  if (!direct_indices.empty()) {
    char_config.wirelength_indices = std::move(direct_indices);
  }
  return char_config;
}

}  // namespace

auto SourceTrunkSegment::build(const Input& input, const Config& config) -> Build
{
  LOG_FATAL_IF(input.source_net == nullptr) << "SourceTrunkSegment build requires an explicit source net.";
  LOG_FATAL_IF(input.reporter == nullptr) << "SourceTrunkSegment build requires an explicit reporter.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "SourceTrunkSegment build requires an explicit Wrapper.";
  auto& source_net = *input.source_net;
  auto* source = input.source;
  auto* sink = input.sink;
  auto& reporter = *input.reporter;

  Build result;
  if (source == nullptr || sink == nullptr) {
    result.summary.failure_reason = "null_source_or_sink_pin";
    LOG_ERROR << "SourceTrunkSegment: source-to-root build failed because source or sink pin is null.";
    EmitSegmentSummary(result, input, config);
    return result;
  }

  const int distance_dbu = geometry::Manhattan(source->get_location(), sink->get_location());
  if (distance_dbu <= 0) {
    ConnectNet(source_net, source, {sink});
    result.summary.success = true;
    EmitSegmentSummary(result, input, config);
    return result;
  }

  LOG_FATAL_IF(input.characterization_input.fast_sta == nullptr)
      << "SourceTrunkSegment build requires explicit FastSTA characterization context.";
  LOG_FATAL_IF(input.dbu_per_um <= 0) << "SourceTrunkSegment build requires explicit positive DBU-per-micron input.";
  const int32_t dbu_per_um = input.dbu_per_um;
  LOG_FATAL_IF(dbu_per_um <= 0) << "SourceTrunkSegment: source-to-root build failed because DBU-per-micron is unavailable.";
  result.summary.length_um = static_cast<double>(distance_dbu) / static_cast<double>(dbu_per_um);

  if (input.required_load_cap_pf <= 0.0) {
    result.summary.failure_reason = "unresolved_required_load_cap";
    LOG_ERROR << "SourceTrunkSegment: source-to-root segment requires a positive root-input load cap.";
    EmitSegmentSummary(result, input, config);
    return result;
  }
  if (input.source_drive_cap_pf <= 0.0) {
    result.summary.failure_reason = "unresolved_source_drive_cap";
    LOG_ERROR << "SourceTrunkSegment: source-to-root segment requires a positive source drive cap.";
    EmitSegmentSummary(result, input, config);
    return result;
  }

  CharacterizationLibrary local_char_library;
  auto* char_library = input.characterization_library == nullptr ? &local_char_library : input.characterization_library;
  const std::vector<double> requested_lengths_um{result.summary.length_um};
  {
    auto char_stage = reporter.beginStage("SourceTrunkSegment", "Ensure characterization",
                                          {
                                              {"length_um", std::to_string(result.summary.length_um)},
                                              {"library_ready", char_library->isReady() ? "true" : "false"},
                                          },
                                          DetailStageReportOptions());
    if (!char_library->isReady()) {
      const auto ensure_result
          = char_library->ensure(input.characterization_input, ConfigureCharConfig(input.characterization_config, requested_lengths_um));
      if (!ensure_result.success) {
        result.summary.failure_reason
            = ensure_result.failure_reason.empty() ? "characterization_library_failed" : ensure_result.failure_reason;
        char_stage.failed({{"reason", result.summary.failure_reason}});
        EmitSegmentSummary(result, input, config);
        return result;
      }
    }
    char_stage.finished();
  }
  const auto& char_builder = char_library->getCharBuilder();
  if (char_builder.get_segment_chars().empty() || char_builder.get_wirelength_unit_um() <= 0.0) {
    result.summary.failure_reason = "no_usable_segment_chars";
    EmitSegmentSummary(result, input, config);
    return result;
  }

  result.summary.length_idx = char_builder.get_length_lattice().coveringIndex(result.summary.length_um);
  result.summary.required_load_cap_idx
      = htree::CoveringBoundaryIndex(input.required_load_cap_pf, char_builder.get_cap_lattice()).value_or(0U);
  result.summary.source_drive_cap_idx
      = htree::CoveringBoundaryIndex(input.source_drive_cap_pf, char_builder.get_cap_lattice()).value_or(0U);
  if (config.min_input_slew_ns.has_value()) {
    result.summary.min_input_slew_idx = htree::CoveringBoundaryIndex(*config.min_input_slew_ns, char_builder.get_slew_lattice());
  }

  if (result.summary.length_idx == 0U || result.summary.required_load_cap_idx == 0U || result.summary.source_drive_cap_idx == 0U) {
    result.summary.failure_reason = "unresolved_segment_boundary_indices";
    EmitSegmentSummary(result, input, config);
    return result;
  }
  if (result.summary.required_load_cap_idx > char_builder.get_cap_steps()) {
    result.summary.failure_reason = "segment_hard_boundary_out_of_range";
    EmitSegmentSummary(result, input, config);
    return result;
  }

  htree::BufferPatternLibrary pattern_library(*input.wrapper);
  htree::SegmentFrontierCatalog segment_frontier_catalog;
  const std::vector<SegmentChar>* all_frontier_entries = nullptr;
  {
    auto frontier_stage = reporter.beginStage("SourceTrunkSegment", "Synthesize segment frontier",
                                              {
                                                  {"length_idx", std::to_string(result.summary.length_idx)},
                                                  {"segment_chars", std::to_string(char_builder.get_segment_chars().size())},
                                              },
                                              DetailStageReportOptions());
    for (const auto& pattern : char_builder.get_buffering_patterns()) {
      pattern_library.add(pattern);
    }
    const htree::RequiredSegmentFrontiers required_segment_frontiers{
        .required_length_indices = {result.summary.length_idx},
        .required_kinds = htree::SegmentFrontierKindSet::allOnly(),
    };
    segment_frontier_catalog
        = htree::SynthesizeSegmentFrontiers(char_builder.get_segment_chars(), pattern_library, required_segment_frontiers);
    all_frontier_entries = segment_frontier_catalog.find(result.summary.length_idx, htree::SegmentFrontierKind::kAll);
    if (all_frontier_entries == nullptr || all_frontier_entries->empty()) {
      result.summary.failure_reason = "missing_required_segment_frontier";
      frontier_stage.failed({{"reason", result.summary.failure_reason}});
      EmitSegmentSummary(result, input, config);
      return result;
    }
    frontier_stage.finished({
        {"length_sets", std::to_string(segment_frontier_catalog.lengthCount())},
        {"frontier_entries", std::to_string(all_frontier_entries->size())},
    });
  }

  {
    auto selection_stage
        = reporter.beginStage("SourceTrunkSegment", "Select segment candidate",
                              {
                                  {"frontier_entries", std::to_string(all_frontier_entries == nullptr ? 0U : all_frontier_entries->size())},
                                  {"required_load_cap_idx", std::to_string(result.summary.required_load_cap_idx)},
                                  {"source_drive_cap_idx", std::to_string(result.summary.source_drive_cap_idx)},
                              },
                              DetailStageReportOptions());
    auto strict_entries = FilterSegmentEntries(*all_frontier_entries, result.summary.required_load_cap_idx,
                                               result.summary.source_drive_cap_idx, result.summary.min_input_slew_idx);
    result.summary.strict_candidate_count = strict_entries.size();
    result.output.best_char = SelectBestSegmentEntry(strict_entries);
    if (!result.output.best_char.has_value() && result.summary.min_input_slew_idx.has_value()) {
      auto relaxed_entries = FilterSegmentEntries(*all_frontier_entries, result.summary.required_load_cap_idx,
                                                  result.summary.source_drive_cap_idx, std::nullopt);
      result.summary.relaxed_candidate_count = relaxed_entries.size();
      result.output.best_char = SelectBestSegmentEntry(relaxed_entries);
      if (result.output.best_char.has_value()) {
        result.summary.used_boundary_relaxation = true;
        result.summary.boundary_relaxation_reason = "dropped_soft_input_slew_boundary";
      }
    }
    if (result.output.best_char.has_value()) {
      selection_stage.finished({
          {"strict_candidates", std::to_string(result.summary.strict_candidate_count)},
          {"relaxed_candidates", std::to_string(result.summary.relaxed_candidate_count)},
          {"used_boundary_relaxation", result.summary.used_boundary_relaxation ? "true" : "false"},
          {"selected_pattern_id", std::to_string(result.output.best_char->get_pattern_id().pack())},
      });
    } else {
      selection_stage.failed({{"reason", "no_hard_boundary_legal_segment_candidate"}});
    }
  }
  if (!result.output.best_char.has_value()) {
    result.summary.failure_reason = "no_hard_boundary_legal_segment_candidate";
    EmitSegmentSummary(result, input, config);
    return result;
  }

  const auto* selected_pattern = pattern_library.find(result.output.best_char->get_pattern_id());
  if (selected_pattern == nullptr) {
    result.summary.failure_reason = "missing_selected_segment_pattern";
    EmitSegmentSummary(result, input, config);
    return result;
  }
  {
    auto object_stage = reporter.beginStage("SourceTrunkSegment", "Build segment objects",
                                            {
                                                {"selected_pattern_id", std::to_string(result.output.best_char->get_pattern_id().pack())},
                                            },
                                            DetailStageReportOptions());
    result.summary.success = BuildSourceTrunkSegmentObjects(result, source_net, source, sink, *selected_pattern, input);
    if (result.summary.success) {
      object_stage.finished({
          {"inserted_insts", std::to_string(result.output.inserted_insts.size())},
          {"inserted_nets", std::to_string(result.output.inserted_nets.size())},
      });
    } else {
      object_stage.failed(
          {{"reason", result.summary.failure_reason.empty() ? "segment_object_build_failed" : result.summary.failure_reason}});
    }
  }
  EmitSegmentSummary(result, input, config);
  return result;
}

}  // namespace icts
