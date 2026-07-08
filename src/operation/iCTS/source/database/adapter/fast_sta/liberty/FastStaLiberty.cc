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
 * @file FastStaLiberty.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Fast STA Liberty timing and power record extraction implementation.
 */

#include "FastStaLiberty.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Type.hh"
#include "Vector.hh"
#include "io/Wrapper.hh"
#include "liberty/Lib.hh"

namespace icts {
namespace {

constexpr double kMilliwattToWatt = 1.0 / 1000.0;

auto convertLibCapToPf(idb::LibCell* lib_cell, double cap_value) -> double
{
  auto* owner_lib = lib_cell != nullptr ? lib_cell->get_owner_lib() : nullptr;
  if (owner_lib == nullptr) {
    return cap_value;
  }
  return idb::ConvertCapUnit(owner_lib->get_cap_unit(), idb::CapacitiveUnit::kPF, cap_value);
}

auto convertLibTimeToNs(idb::LibCell* lib_cell, double time_value) -> double
{
  auto* owner_lib = lib_cell != nullptr ? lib_cell->get_owner_lib() : nullptr;
  if (owner_lib == nullptr) {
    return time_value;
  }
  return owner_lib->convert_time_unit_to_ns(time_value);
}

auto queryLibPortCapacitancePf(idb::LibCell* lib_cell, idb::LibPort* lib_port) -> double
{
  if (lib_cell == nullptr || lib_port == nullptr || lib_port->isInput() == 0U) {
    return 0.0;
  }

  double cap_value = lib_port->get_port_cap();
  cap_value = std::max(cap_value, lib_port->get_port_cap(idb::AnalysisMode::kMax, idb::TransType::kRise).value_or(0.0));
  cap_value = std::max(cap_value, lib_port->get_port_cap(idb::AnalysisMode::kMax, idb::TransType::kFall).value_or(0.0));
  cap_value = std::max(cap_value, lib_port->get_port_cap(idb::AnalysisMode::kMin, idb::TransType::kRise).value_or(0.0));
  cap_value = std::max(cap_value, lib_port->get_port_cap(idb::AnalysisMode::kMin, idb::TransType::kFall).value_or(0.0));
  return convertLibCapToPf(lib_cell, cap_value);
}

auto findBufferArcSet(idb::LibCell* lib_cell) -> std::optional<idb::LibArcSet*>
{
  if (lib_cell == nullptr) {
    return std::nullopt;
  }

  idb::LibPort* input = nullptr;
  idb::LibPort* output = nullptr;
  lib_cell->bufferPorts(input, output);
  if (input == nullptr || output == nullptr) {
    return std::nullopt;
  }

  auto timing_arc_set = lib_cell->findLibertyArcSet(input->get_port_name(), output->get_port_name(), idb::LibArc::TimingType::kComb);
  if (!timing_arc_set.has_value()) {
    timing_arc_set = lib_cell->findLibertyArcSet(input->get_port_name(), output->get_port_name());
  }
  return timing_arc_set;
}

auto toFastStaAxisKind(idb::LibLutTableTemplate::Variable variable) -> FastStaLibertyAxisKind
{
  switch (variable) {
    case idb::LibLutTableTemplate::Variable::INPUT_NET_TRANSITION:
    case idb::LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION:
    case idb::LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME:
      return FastStaLibertyAxisKind::kInputSlew;
    case idb::LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE:
    case idb::LibLutTableTemplate::Variable::EQUAL_OR_OPPOSITE_OUTPUT_NET_CAPACITANCE:
      return FastStaLibertyAxisKind::kOutputLoad;
    default:
      return FastStaLibertyAxisKind::kUnknown;
  }
}

auto convertTableAxisValue(idb::LibCell* lib_cell, idb::LibLutTableTemplate::Variable variable, double value) -> double
{
  auto* owner_lib = lib_cell != nullptr ? lib_cell->get_owner_lib() : nullptr;
  if (owner_lib == nullptr) {
    return value;
  }

  switch (toFastStaAxisKind(variable)) {
    case FastStaLibertyAxisKind::kInputSlew:
      return owner_lib->convert_time_unit_to_ns(value);
    case FastStaLibertyAxisKind::kOutputLoad:
      return convertLibCapToPf(lib_cell, value);
    case FastStaLibertyAxisKind::kUnknown:
      return value;
  }
  return value;
}

auto convertTableValue(idb::LibCell* lib_cell, FastStaLibertyTableKind kind, double value) -> double
{
  switch (kind) {
    case FastStaLibertyTableKind::kCellDelay:
    case FastStaLibertyTableKind::kOutputSlew:
      return convertLibTimeToNs(lib_cell, value);
    case FastStaLibertyTableKind::kInternalPower:
      return lib_cell != nullptr ? lib_cell->convertInternalPowerTableToMwNs(value) : value;
  }
  return value;
}

auto appendAxis(idb::LibCell* lib_cell, idb::LibTable* table, std::optional<idb::LibLutTableTemplate::Variable> variable,
                unsigned axis_index, FastStaLibertyTable& table_record) -> void
{
  if (!variable.has_value() || axis_index >= table->getAxesSize()) {
    return;
  }

  FastStaLibertyAxis axis;
  axis.kind = toFastStaAxisKind(*variable);
  auto& source_axis_values = table->getAxis(axis_index).get_axis_values();
  axis.values.reserve(source_axis_values.size());
  for (const auto& value : source_axis_values) {
    if (value == nullptr) {
      continue;
    }
    axis.values.push_back(convertTableAxisValue(lib_cell, *variable, value->getFloatValue()));
  }
  table_record.axes.push_back(std::move(axis));
}

auto extractTable(idb::LibCell* lib_cell, idb::LibTable* table, FastStaLibertyTableKind kind, FastStaTransition transition)
    -> FastStaLibertyTable
{
  FastStaLibertyTable table_record;
  table_record.kind = kind;
  table_record.transition = transition;
  if (table == nullptr) {
    return table_record;
  }

  auto* table_template = table->get_table_template();
  if (table_template != nullptr) {
    appendAxis(lib_cell, table, table_template->get_template_variable1(), 0U, table_record);
    appendAxis(lib_cell, table, table_template->get_template_variable2(), 1U, table_record);
  }

  const auto& source_values = table->get_table_values();
  table_record.values.reserve(source_values.size());
  for (const auto& value : source_values) {
    if (value == nullptr) {
      continue;
    }
    table_record.values.push_back(convertTableValue(lib_cell, kind, value->getFloatValue()));
  }
  return table_record;
}

auto appendDelayTable(idb::LibCell* lib_cell, idb::LibDelayTableModel* delay_model, idb::LibTable::TableType table_type,
                      FastStaLibertyTableKind kind, FastStaTransition transition, std::vector<FastStaLibertyTable>& tables) -> void
{
  auto* table = delay_model != nullptr ? delay_model->getTable(static_cast<int>(table_type)) : nullptr;
  if (table == nullptr) {
    return;
  }
  tables.push_back(extractTable(lib_cell, table, kind, transition));
}

auto appendPowerTable(idb::LibCell* lib_cell, idb::LibPowerTableModel* power_model, idb::LibTable::TableType table_type,
                      FastStaTransition transition, std::vector<FastStaLibertyTable>& tables) -> void
{
  auto* table = power_model != nullptr ? power_model->getTable(CAST_POWER_TYPE_TO_INDEX(table_type)) : nullptr;
  if (table == nullptr) {
    return;
  }
  tables.push_back(extractTable(lib_cell, table, FastStaLibertyTableKind::kInternalPower, transition));
}

auto calcLeakagePowerW(idb::LibCell* lib_cell) -> double
{
  if (lib_cell == nullptr) {
    return 0.0;
  }

  const double cell_leakage_mw = lib_cell->get_cell_leakage_power();
  if (std::isfinite(cell_leakage_mw) && cell_leakage_mw > 0.0) {
    return cell_leakage_mw * kMilliwattToWatt;
  }

  double leakage_mw = 0.0;
  for (auto* leakage_power : lib_cell->getLeakagePowerList()) {
    if (leakage_power != nullptr && leakage_power->get_when().empty() && std::isfinite(leakage_power->get_value())) {
      leakage_mw += leakage_power->get_value();
    }
  }
  return leakage_mw * kMilliwattToWatt;
}

auto percentOrDefault(double value, double default_value) -> double
{
  return value > 0.0 && value < 1.0 ? value : default_value;
}

auto findBestTimingArc(idb::LibArcSet* arc_set) -> idb::LibArc*
{
  if (arc_set == nullptr) {
    return nullptr;
  }
  idb::LibArc* first_enabled = nullptr;
  for (const auto& arc_holder : arc_set->get_arcs()) {
    auto* arc = arc_holder.get();
    if (arc == nullptr || arc->isDisableArc() != 0U) {
      continue;
    }
    if (first_enabled == nullptr) {
      first_enabled = arc;
    }
    if (arc->get_when().empty()) {
      return arc;
    }
  }
  return first_enabled;
}

auto appendPowerArcTables(idb::LibCell* lib_cell, idb::LibPowerArcSet* power_arc_set, FastStaLibertyArc& arc_record) -> void
{
  if (power_arc_set == nullptr) {
    return;
  }
  for (const auto& power_arc_holder : power_arc_set->get_power_arcs()) {
    auto* power_arc = power_arc_holder.get();
    auto* internal_power
        = power_arc != nullptr && power_arc->get_internal_power_info() != nullptr ? power_arc->get_internal_power_info().get() : nullptr;
    auto* power_model
        = internal_power != nullptr ? dynamic_cast<idb::LibPowerTableModel*>(internal_power->get_power_table_model()) : nullptr;
    if (power_model == nullptr) {
      continue;
    }
    appendPowerTable(lib_cell, power_model, idb::LibTable::TableType::kRisePower, FastStaTransition::kRise,
                     arc_record.internal_power_tables);
    appendPowerTable(lib_cell, power_model, idb::LibTable::TableType::kFallPower, FastStaTransition::kFall,
                     arc_record.internal_power_tables);
  }
}

auto extractBufferCellFromLibCell(Wrapper& wrapper, idb::LibCell* lib_cell) -> FastStaLibertyCell
{
  if (lib_cell == nullptr) {
    return FastStaLibertyCell{};
  }

  idb::LibPort* input_port = nullptr;
  idb::LibPort* output_port = nullptr;
  lib_cell->bufferPorts(input_port, output_port);
  const auto cell_master = std::string(lib_cell->get_cell_name());
  const auto input_port_name = input_port != nullptr ? std::string(input_port->get_port_name()) : std::string{};
  const auto output_port_name = output_port != nullptr ? std::string(output_port->get_port_name()) : std::string{};
  auto* owner_lib = lib_cell->get_owner_lib();

  auto output_cap_limit_pf = 0.0;
  if (output_port != nullptr) {
    if (auto cap_limit = output_port->get_port_cap_limit(idb::AnalysisMode::kMax); cap_limit.has_value()) {
      output_cap_limit_pf = convertLibCapToPf(lib_cell, *cap_limit);
    }
  }
  if (output_cap_limit_pf <= 0.0) {
    output_cap_limit_pf = wrapper.queryCellOutPinCapTableAxisMax(cell_master);
  }

  auto input_slew_limit_ns = 0.0;
  if (input_port != nullptr) {
    if (auto slew_limit = input_port->get_port_slew_limit(idb::AnalysisMode::kMax); slew_limit.has_value()) {
      input_slew_limit_ns = convertLibTimeToNs(lib_cell, *slew_limit);
    }
  }
  if (input_slew_limit_ns <= 0.0) {
    input_slew_limit_ns = wrapper.queryCellInPinSlewTableAxisMax(cell_master);
  }

  FastStaLibertyCell cell{
      .cell_master = cell_master,
      .input_port = input_port_name,
      .output_port = output_port_name,
      .input_cap_pf = queryLibPortCapacitancePf(lib_cell, input_port),
      .output_cap_limit_pf = output_cap_limit_pf,
      .input_slew_limit_ns = input_slew_limit_ns,
      .input_threshold_rise = owner_lib != nullptr ? percentOrDefault(owner_lib->get_input_threshold_pct_rise(), 0.5) : 0.5,
      .input_threshold_fall = owner_lib != nullptr ? percentOrDefault(owner_lib->get_input_threshold_pct_fall(), 0.5) : 0.5,
      .output_threshold_rise = owner_lib != nullptr ? percentOrDefault(owner_lib->get_output_threshold_pct_rise(), 0.5) : 0.5,
      .output_threshold_fall = owner_lib != nullptr ? percentOrDefault(owner_lib->get_output_threshold_pct_fall(), 0.5) : 0.5,
      .slew_lower_threshold_rise = owner_lib != nullptr ? percentOrDefault(owner_lib->get_slew_lower_threshold_pct_rise(), 0.3) : 0.3,
      .slew_lower_threshold_fall = owner_lib != nullptr ? percentOrDefault(owner_lib->get_slew_lower_threshold_pct_fall(), 0.3) : 0.3,
      .slew_upper_threshold_rise = owner_lib != nullptr ? percentOrDefault(owner_lib->get_slew_upper_threshold_pct_rise(), 0.7) : 0.7,
      .slew_upper_threshold_fall = owner_lib != nullptr ? percentOrDefault(owner_lib->get_slew_upper_threshold_pct_fall(), 0.7) : 0.7,
      .slew_derate_from_library = owner_lib != nullptr && owner_lib->get_slew_derate_from_library() > 0.0
                                      ? owner_lib->get_slew_derate_from_library()
                                      : 1.0,
      .area_um2 = std::max(0.0, wrapper.queryCellAreaUm2(cell_master)),
      .voltage_v = owner_lib != nullptr ? owner_lib->get_nom_voltage() : 0.0,
      .leakage_power_w = calcLeakagePowerW(lib_cell),
      .timing_arc = FastStaLibertyArc{
          .from_port = input_port_name,
          .to_port = output_port_name,
          .delay_tables = {},
          .slew_tables = {},
          .internal_power_tables = {},
      },
  };

  auto timing_arc_set = findBufferArcSet(lib_cell);
  auto* timing_arc = findBestTimingArc(timing_arc_set.value_or(nullptr));
  auto* delay_model = timing_arc != nullptr ? dynamic_cast<idb::LibDelayTableModel*>(timing_arc->get_table_model()) : nullptr;
  if (timing_arc != nullptr) {
    cell.timing_arc.negative_unate = timing_arc->isNegativeArc() != 0U;
  }
  if (delay_model != nullptr) {
    appendDelayTable(lib_cell, delay_model, idb::LibTable::TableType::kCellRise, FastStaLibertyTableKind::kCellDelay,
                     FastStaTransition::kRise, cell.timing_arc.delay_tables);
    appendDelayTable(lib_cell, delay_model, idb::LibTable::TableType::kCellFall, FastStaLibertyTableKind::kCellDelay,
                     FastStaTransition::kFall, cell.timing_arc.delay_tables);
    appendDelayTable(lib_cell, delay_model, idb::LibTable::TableType::kRiseTransition, FastStaLibertyTableKind::kOutputSlew,
                     FastStaTransition::kRise, cell.timing_arc.slew_tables);
    appendDelayTable(lib_cell, delay_model, idb::LibTable::TableType::kFallTransition, FastStaLibertyTableKind::kOutputSlew,
                     FastStaTransition::kFall, cell.timing_arc.slew_tables);
  }

  auto power_arc_set = lib_cell->findLibertyPowerArcSet(input_port_name.c_str(), output_port_name.c_str());
  appendPowerArcTables(lib_cell, power_arc_set.value_or(nullptr), cell.timing_arc);
  return cell;
}

}  // namespace

auto FastStaLiberty::extractBufferCell(Wrapper& wrapper, const std::string& cell_master) -> FastStaLibertyCell
{
  auto* lib_cell = wrapper.findLibertyCell(cell_master);
  if (lib_cell != nullptr) {
    return extractBufferCellFromLibCell(wrapper, lib_cell);
  }

  auto [input_port, output_port] = wrapper.queryBufferPorts(cell_master);
  auto output_cap_limit_pf = wrapper.queryCellOutPinCapLimit(cell_master);
  if (output_cap_limit_pf <= 0.0) {
    output_cap_limit_pf = wrapper.queryCellOutPinCapTableAxisMax(cell_master);
  }
  auto input_slew_limit_ns = wrapper.queryCellInPinSlewLimit(cell_master);
  if (input_slew_limit_ns <= 0.0) {
    input_slew_limit_ns = wrapper.queryCellInPinSlewTableAxisMax(cell_master);
  }
  return FastStaLibertyCell{
      .cell_master = cell_master,
      .input_port = input_port,
      .output_port = output_port,
      .input_cap_pf = wrapper.queryCharInputPinCap(cell_master),
      .output_cap_limit_pf = output_cap_limit_pf,
      .input_slew_limit_ns = input_slew_limit_ns,
      .input_threshold_rise = 0.5,
      .input_threshold_fall = 0.5,
      .output_threshold_rise = 0.5,
      .output_threshold_fall = 0.5,
      .slew_lower_threshold_rise = 0.3,
      .slew_lower_threshold_fall = 0.3,
      .slew_upper_threshold_rise = 0.7,
      .slew_upper_threshold_fall = 0.7,
      .slew_derate_from_library = 1.0,
      .area_um2 = std::max(0.0, wrapper.queryCellAreaUm2(cell_master)),
      .voltage_v = 0.0,
      .leakage_power_w = 0.0,
      .timing_arc = FastStaLibertyArc{
          .from_port = input_port,
          .to_port = output_port,
          .delay_tables = {},
          .slew_tables = {},
          .internal_power_tables = {},
      },
  };
}

}  // namespace icts
