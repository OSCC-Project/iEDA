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
 * @file WrapperLiberty.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-29
 * @brief Wrapper-backed Liberty queries for iCTS.
 */

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbLayout.h"
#include "IdbUnits.h"
// use iEDA Rust liberty parser
#include "Log.hh"
#include "Type.hh"
#include "Vector.hh"
#include "Wrapper.hh"
#include "config/Config.hh"
#include "design/Design.hh"
#include "design/Inst.hh"
#include "design/Pin.hh"
#include "dm_config.h"
#include "idm.h"
#include "liberty/Lib.hh"

namespace icts {
namespace {

constexpr double kMilliwattToWatt = 1.0 / 1000.0;
constexpr double kClockToggleDensityNumerator = 2.0;
constexpr std::array<idb::LibTable::TableType, 4> kCharArcTableTypes = {
    idb::LibTable::TableType::kCellRise,
    idb::LibTable::TableType::kCellFall,
    idb::LibTable::TableType::kRiseTransition,
    idb::LibTable::TableType::kFallTransition,
};

auto normalizePortName(const std::string& pin_name) -> std::string
{
  const auto separator_pos = pin_name.rfind('/');
  return separator_pos == std::string::npos ? pin_name : pin_name.substr(separator_pos + 1);
}

auto makeCharQueryContext(const char* query_name, const std::string& cell_master) -> std::string
{
  return "Characterization query [" + std::string(query_name) + "] for liberty cell \"" + cell_master + "\"";
}

auto resolvePositiveMax(const std::vector<std::optional<double>>& values) -> double
{
  double max_value = 0.0;
  for (const auto& value : values) {
    if (!value.has_value() || *value <= 0.0) {
      continue;
    }
    max_value = std::max(max_value, *value);
  }
  return max_value;
}

auto convertLibCapToPf(idb::LibCell* lib_cell, double cap_value) -> double
{
  // LibPort caps are already normalized to pF by the Rust liberty parser
  // (LUT table axes are NOT normalized and use ConvertCapUnit directly elsewhere).
  (void)lib_cell;
  return cap_value;
}

auto convertLibTimeToNs(idb::LibCell* lib_cell, double time_value) -> double
{
  auto* owner_lib = lib_cell != nullptr ? lib_cell->get_owner_lib() : nullptr;
  if (owner_lib == nullptr) {
    return time_value;
  }
  return owner_lib->convert_time_unit_to_ns(time_value);
}

auto convertPfLoadToLibUnit(idb::LibCell* lib_cell, double load_pf) -> double
{
  auto* owner_lib = lib_cell != nullptr ? lib_cell->get_owner_lib() : nullptr;
  if (owner_lib == nullptr) {
    return load_pf;
  }

  if (owner_lib->get_cap_unit() == idb::CapacitiveUnit::kFF) {
    return PF_TO_FF(load_pf);
  }
  if (owner_lib->get_cap_unit() == idb::CapacitiveUnit::kPF) {
    return load_pf;
  }
  if (owner_lib->get_cap_unit() == idb::CapacitiveUnit::kF) {
    return PF_TO_F(load_pf);
  }
  return load_pf;
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

auto convertAxisValue(idb::LibLibrary* owner_lib, idb::LibLutTableTemplate::Variable variable, double axis_value) -> double
{
  if (owner_lib == nullptr) {
    return 0.0;
  }

  switch (variable) {
    case idb::LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE:
    case idb::LibLutTableTemplate::Variable::EQUAL_OR_OPPOSITE_OUTPUT_NET_CAPACITANCE:
      return idb::ConvertCapUnit(owner_lib->get_cap_unit(), idb::CapacitiveUnit::kPF, axis_value);
    case idb::LibLutTableTemplate::Variable::INPUT_NET_TRANSITION:
    case idb::LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION:
    case idb::LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME:
    case idb::LibLutTableTemplate::Variable::CONSTRAINED_PIN_TRANSITION:
      return owner_lib->convert_time_unit_to_ns(axis_value);
    default:
      return 0.0;
  }
}

auto queryBufferTableAxisMax(idb::LibCell* lib_cell, const char* query_name,
                             std::initializer_list<idb::LibLutTableTemplate::Variable> target_variables) -> double
{
  if (lib_cell == nullptr) {
    return 0.0;
  }
  const std::string cell_master = lib_cell->get_cell_name();

  auto timing_arc_set = findBufferArcSet(lib_cell);
  if (!timing_arc_set.has_value() || timing_arc_set.value() == nullptr || timing_arc_set.value()->get_arcs().empty()) {
    LOG_WARNING << makeCharQueryContext(query_name, cell_master) << " failed: no buffer timing arcs are available; return 0.0.";
    return 0.0;
  }

  auto* owner_lib = lib_cell->get_owner_lib();
  LOG_WARNING_IF(owner_lib == nullptr) << makeCharQueryContext(query_name, cell_master) << " failed: owner liberty is null; return 0.0.";
  if (owner_lib == nullptr) {
    return 0.0;
  }

  auto is_target_variable = [&target_variables](idb::LibLutTableTemplate::Variable variable) -> bool {
    return std::ranges::find(target_variables, variable) != target_variables.end();
  };

  double min_axis_max = std::numeric_limits<double>::infinity();
  bool found_axis = false;

  for (const auto& timing_arc_holder : timing_arc_set.value()->get_arcs()) {
    auto* timing_arc = timing_arc_holder.get();
    auto* delay_model = timing_arc != nullptr ? dynamic_cast<idb::LibDelayTableModel*>(timing_arc->get_table_model()) : nullptr;
    if (delay_model == nullptr) {
      continue;
    }

    for (const auto table_type : kCharArcTableTypes) {
      auto* table = delay_model->getTable(static_cast<int>(table_type));
      if (table == nullptr || table->getAxesSize() == 0U) {
        continue;
      }

      auto* table_template = table->get_table_template();
      if (table_template == nullptr) {
        continue;
      }

      auto inspect_axis = [&](std::optional<idb::LibLutTableTemplate::Variable> variable, unsigned axis_index) -> void {
        if (!variable.has_value() || !is_target_variable(*variable) || axis_index >= table->getAxesSize()) {
          return;
        }

        auto& axis_values = table->getAxis(axis_index).get_axis_values();
        if (axis_values.empty()) {
          return;
        }

        const double converted_axis_max = convertAxisValue(owner_lib, *variable, axis_values.back()->getFloatValue());
        if (converted_axis_max > 0.0) {
          min_axis_max = std::min(min_axis_max, converted_axis_max);
          found_axis = true;
        }
      };

      inspect_axis(table_template->get_template_variable1(), 0U);
      inspect_axis(table_template->get_template_variable2(), 1U);
    }
  }

  if (!found_axis) {
    LOG_WARNING << makeCharQueryContext(query_name, cell_master) << " found no matching non-empty table axis values; return 0.0.";
    return 0.0;
  }

  return min_axis_max;
}

auto queryLibOutputPinCapLimitPf(idb::LibCell* lib_cell, const Pin* pin) -> double
{
  if (pin == nullptr || pin->get_inst() == nullptr) {
    return 0.0;
  }
  const auto* inst = pin->get_inst();
  const auto& cell_master = inst->get_cell_master();
  const auto pin_full_name = Design::getPinFullName(pin);
  if (lib_cell == nullptr) {
    LOG_WARNING << "Clock-source drive-cap query skipped: liberty cell \"" << cell_master << "\" is not found for " << pin_full_name << ".";
    return 0.0;
  }

  const auto port_name = normalizePortName(pin->get_name());
  auto* lib_port = lib_cell->get_cell_port_or_port_bus(port_name.c_str());
  if (lib_port == nullptr) {
    LOG_WARNING << "Clock-source drive-cap query skipped: liberty port \"" << port_name << "\" is not found on cell " << cell_master << ".";
    return 0.0;
  }
  if (lib_port->isOutput() == 0U) {
    LOG_WARNING << "Clock-source drive-cap query skipped: liberty port \"" << port_name << "\" on cell " << cell_master
                << " is not an output/inout port.";
    return 0.0;
  }

  const auto cap_limit = lib_port->get_port_cap_limit(idb::AnalysisMode::kMax);
  if (!cap_limit.has_value() || *cap_limit <= 0.0) {
    return 0.0;
  }
  return convertLibCapToPf(lib_cell, *cap_limit);
}

auto queryConfiguredMaxCapBoundaryPf(std::optional<double> configured_max_cap_pf, const Pin* clock_source) -> double
{
  if (configured_max_cap_pf.has_value() && *configured_max_cap_pf > 0.0) {
    LOG_WARNING << "Clock-source drive-cap query uses configured max_cap=" << *configured_max_cap_pf
                << " pF as the hard source boundary for " << Design::getPinFullName(clock_source)
                << " because source-specific Liberty cap limit is unavailable.";
    return *configured_max_cap_pf;
  }
  return 0.0;
}

auto flipTrans(idb::TransType trans_type) -> idb::TransType
{
  return trans_type == idb::TransType::kRise ? idb::TransType::kFall : idb::TransType::kRise;
}

auto makeInvalidCost(const std::string& method, const std::string& cell_master, double input_slew_ns, double output_load_pf)
    -> Wrapper::RootDriverCost
{
  Wrapper::RootDriverCost cost;
  cost.method = method;
  cost.cell_master = cell_master;
  cost.input_slew_ns = input_slew_ns;
  cost.output_load_pf = output_load_pf;
  return cost;
}

auto lookupRootCellDelayNs(idb::LibCell* lib_cell, idb::LibArcSet* timing_arc_set, double input_slew_ns, double output_load_pf) -> double
{
  if (lib_cell == nullptr || timing_arc_set == nullptr || input_slew_ns < 0.0 || output_load_pf < 0.0) {
    return 0.0;
  }

  const double output_load = convertPfLoadToLibUnit(lib_cell, output_load_pf);
  double worst_delay_ns = 0.0;
  bool has_delay = false;
  for (const auto input_trans : {idb::TransType::kRise, idb::TransType::kFall}) {
    const auto output_trans = timing_arc_set->isNegativeArc() != 0U ? flipTrans(input_trans) : input_trans;
    if (!timing_arc_set->isMatchTimingType(output_trans)) {
      continue;
    }
    const auto delay_values = timing_arc_set->getDelayOrConstrainCheckNs(input_trans, output_trans, input_slew_ns, output_load);
    for (const double delay_ns : delay_values) {
      if (!std::isfinite(delay_ns)) {
        continue;
      }
      worst_delay_ns = has_delay ? std::max(worst_delay_ns, delay_ns) : delay_ns;
      has_delay = true;
    }
  }
  return has_delay ? worst_delay_ns : 0.0;
}

auto lookupRootCellOutputSlewNs(idb::LibCell* lib_cell, idb::LibArcSet* timing_arc_set, double input_slew_ns, double output_load_pf)
    -> double
{
  if (lib_cell == nullptr || timing_arc_set == nullptr || input_slew_ns < 0.0 || output_load_pf < 0.0) {
    return 0.0;
  }

  const double output_load = convertPfLoadToLibUnit(lib_cell, output_load_pf);
  double worst_slew_ns = 0.0;
  bool has_slew = false;
  for (const auto input_trans : {idb::TransType::kRise, idb::TransType::kFall}) {
    const auto output_trans = timing_arc_set->isNegativeArc() != 0U ? flipTrans(input_trans) : input_trans;
    if (!timing_arc_set->isMatchTimingType(output_trans)) {
      continue;
    }
    const auto slew_values = timing_arc_set->getSlewNs(input_trans, output_trans, input_slew_ns, output_load);
    for (const double slew_ns : slew_values) {
      if (!std::isfinite(slew_ns)) {
        continue;
      }
      worst_slew_ns = has_slew ? std::max(worst_slew_ns, slew_ns) : slew_ns;
      has_slew = true;
    }
  }
  return has_slew ? worst_slew_ns : 0.0;
}

auto lookupInternalPowerW(idb::LibCell* lib_cell, idb::LibPowerArcSet* power_arc_set, double input_slew_ns, double output_load_pf,
                          double clock_period_ns) -> double
{
  if (lib_cell == nullptr || power_arc_set == nullptr || input_slew_ns < 0.0 || output_load_pf < 0.0 || clock_period_ns <= 0.0) {
    return 0.0;
  }

  const double output_load = convertPfLoadToLibUnit(lib_cell, output_load_pf);
  const double output_toggle_per_ns = kClockToggleDensityNumerator / clock_period_ns;
  double power_mw = 0.0;

  for (const auto& power_arc_ptr : power_arc_set->get_power_arcs()) {
    auto* power_arc = power_arc_ptr.get();
    if (power_arc == nullptr || power_arc->get_internal_power_info() == nullptr) {
      continue;
    }
    auto* internal_power_info = power_arc->get_internal_power_info().get();
    const double rise_energy_mw_ns
        = lib_cell->convertInternalPowerTableToMwNs(internal_power_info->gatePower(idb::TransType::kRise, input_slew_ns, output_load));
    const double fall_energy_mw_ns
        = lib_cell->convertInternalPowerTableToMwNs(internal_power_info->gatePower(idb::TransType::kFall, input_slew_ns, output_load));
    const double average_energy_mw_ns = (rise_energy_mw_ns + fall_energy_mw_ns) / 2.0;
    if (std::isfinite(average_energy_mw_ns)) {
      power_mw += output_toggle_per_ns * average_energy_mw_ns;
    }
  }

  return power_mw * kMilliwattToWatt;
}

auto lookupLeakagePowerW(idb::LibCell* lib_cell) -> double
{
  if (lib_cell == nullptr) {
    return 0.0;
  }

  const double default_leakage_power_w = lib_cell->get_cell_leakage_power();
  if (std::isfinite(default_leakage_power_w) && default_leakage_power_w > 0.0) {
    return default_leakage_power_w;
  }

  double unconditional_leakage_power_w = 0.0;
  for (auto* leakage_power : lib_cell->getLeakagePowerList()) {
    if (leakage_power != nullptr && leakage_power->get_when().empty() && std::isfinite(leakage_power->get_value())) {
      unconditional_leakage_power_w += leakage_power->get_value();
    }
  }
  return unconditional_leakage_power_w;
}

}  // namespace

auto Wrapper::loadLibertyIfNeeded() const -> void
{
  if (_liberty_loaded) {
    return;
  }
  _liberty_loaded = true;
  _lib_libraries.clear();
  _lib_cell_by_master.clear();

  auto& lib_paths = dmInst->get_config().get_lib_paths();
  if (lib_paths.empty()) {
    LOG_WARNING << "Wrapper: no Liberty files are configured; Liberty-backed CTS queries will return empty values.";
    return;
  }

  for (const auto& lib_path : lib_paths) {
    idb::Lib lib;
    auto reader = lib.loadLibertyWithRustParser(lib_path.c_str());
    LOG_FATAL_IF(reader.linkLib() == 0U) << "Wrapper: failed to link Liberty file " << lib_path << ".";
    auto* library_builder = reader.get_library_builder();
    LOG_FATAL_IF(library_builder == nullptr) << "Wrapper: Liberty library builder is null for " << lib_path << ".";
    auto library = library_builder->takeLib();
    LOG_FATAL_IF(library == nullptr) << "Wrapper: Liberty library is null for " << lib_path << ".";
    for (const auto& cell : library->get_cells()) {
      if (cell == nullptr) {
        continue;
      }
      _lib_cell_by_master[cell->get_cell_name()] = cell.get();
    }
    _lib_libraries.push_back(std::move(library));
  }
  LOG_INFO << "Wrapper: loaded " << _lib_libraries.size() << " Liberty file(s) for CTS queries.";
}

auto Wrapper::findLibertyCell(const std::string& cell_master) const -> idb::LibCell*
{
  if (cell_master.empty()) {
    return nullptr;
  }
  loadLibertyIfNeeded();
  const auto iter = _lib_cell_by_master.find(cell_master);
  return iter == _lib_cell_by_master.end() ? nullptr : iter->second;
}

auto Wrapper::queryCellOutPinCapLimit(const std::string& cell_master) const -> double
{
  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << makeCharQueryContext("output pin cap limit", cell_master)
                << " failed: liberty cell not found; caller may use table-axis max policy.";
    return 0.0;
  }

  idb::LibPort* input = nullptr;
  idb::LibPort* output = nullptr;
  lib_cell->bufferPorts(input, output);
  if (output == nullptr) {
    LOG_WARNING << makeCharQueryContext("output pin cap limit", cell_master)
                << " failed: output pin is unavailable; caller may use table-axis max policy.";
    return 0.0;
  }

  auto cap_limit = output->get_port_cap_limit(idb::AnalysisMode::kMax);
  if (!cap_limit.has_value()) {
    LOG_WARNING << makeCharQueryContext("output pin cap limit", cell_master)
                << " failed: max cap limit is not defined on output pin; caller may use table-axis max policy.";
    return 0.0;
  }
  return convertLibCapToPf(lib_cell, *cap_limit);
}

auto Wrapper::queryCellOutPinCapTableAxisMax(const std::string& cell_master) const -> double
{
  return queryBufferTableAxisMax(findLibertyCell(cell_master), "output pin cap table-axis max",
                                 {idb::LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE,
                                  idb::LibLutTableTemplate::Variable::EQUAL_OR_OPPOSITE_OUTPUT_NET_CAPACITANCE});
}

auto Wrapper::queryClockSourceDriveCapLimit(const ClockSourceDriveCapLimitInput& input) const -> double
{
  const auto* clock_source = input.clock_source;
  if (clock_source == nullptr) {
    LOG_WARNING << "Clock-source drive-cap query skipped: clock source pin is null.";
    return 0.0;
  }

  if (auto* inst = clock_source->get_inst(); inst != nullptr) {
    const auto& cell_master = inst->get_cell_master();
    auto* lib_cell = findLibertyCell(cell_master);
    const double lib_cap_limit_pf = queryLibOutputPinCapLimitPf(lib_cell, clock_source);
    if (lib_cap_limit_pf > 0.0) {
      return lib_cap_limit_pf;
    }

    const double table_axis_cap_limit_pf = queryCellOutPinCapTableAxisMax(cell_master);
    if (table_axis_cap_limit_pf > 0.0) {
      return table_axis_cap_limit_pf;
    }

    const double configured_cap_limit_pf = queryConfiguredMaxCapBoundaryPf(input.configured_max_cap_pf, clock_source);
    return configured_cap_limit_pf > 0.0 ? configured_cap_limit_pf : 0.0;
  }

  const double configured_cap_limit_pf = queryConfiguredMaxCapBoundaryPf(input.configured_max_cap_pf, clock_source);
  if (configured_cap_limit_pf > 0.0) {
    return configured_cap_limit_pf;
  }
  if (!clock_source->is_io()) {
    LOG_WARNING << "Clock-source drive-cap query skipped: CTS pin \"" << Design::getPinFullName(clock_source)
                << "\" has no owning inst and is not marked as top-level IO.";
  }
  return 0.0;
}

auto Wrapper::queryClockSourceDriveCapLimit(const Config& config, const Pin* clock_source) const -> double
{
  return queryClockSourceDriveCapLimit(ClockSourceDriveCapLimitInput{
      .clock_source = clock_source,
      .configured_max_cap_pf
      = config.has_max_cap() && config.get_max_cap() > 0.0 ? std::optional<double>{config.get_max_cap()} : std::nullopt,
  });
}

auto Wrapper::queryCellInPinSlewLimit(const std::string& cell_master) const -> double
{
  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << makeCharQueryContext("input pin slew limit", cell_master)
                << " failed: liberty cell not found; caller may use table-axis max policy.";
    return 0.0;
  }

  idb::LibPort* input = nullptr;
  idb::LibPort* output = nullptr;
  lib_cell->bufferPorts(input, output);
  if (input == nullptr) {
    LOG_WARNING << makeCharQueryContext("input pin slew limit", cell_master)
                << " failed: input pin is unavailable; caller may use table-axis max policy.";
    return 0.0;
  }

  auto slew_limit = input->get_port_slew_limit(idb::AnalysisMode::kMax);
  if (!slew_limit.has_value()) {
    LOG_WARNING << makeCharQueryContext("input pin slew limit", cell_master)
                << " failed: max slew limit is not defined on input pin; caller may use table-axis max policy.";
    return 0.0;
  }
  return convertLibTimeToNs(lib_cell, *slew_limit);
}

auto Wrapper::queryCellInPinSlewTableAxisMax(const std::string& cell_master) const -> double
{
  return queryBufferTableAxisMax(
      findLibertyCell(cell_master), "input pin slew table-axis max",
      {idb::LibLutTableTemplate::Variable::INPUT_NET_TRANSITION, idb::LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION,
       idb::LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME, idb::LibLutTableTemplate::Variable::CONSTRAINED_PIN_TRANSITION});
}

auto Wrapper::queryPinSlewLimit(const PinSlewLimitInput& input) const -> double
{
  const auto* pin = input.pin;
  if (pin == nullptr) {
    LOG_WARNING << "Pin-slew-limit query skipped: CTS pin is null.";
    return 0.0;
  }

  auto* inst = pin->get_inst();
  if (inst == nullptr) {
    const double configured_limit_ns = input.configured_max_sink_tran_ns;
    return configured_limit_ns > 0.0 ? configured_limit_ns : 0.0;
  }

  const auto& cell_master = inst->get_cell_master();
  if (cell_master.empty()) {
    const double configured_limit_ns = input.configured_max_sink_tran_ns;
    return configured_limit_ns > 0.0 ? configured_limit_ns : 0.0;
  }

  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << "Pin-slew-limit query skipped: liberty cell \"" << cell_master << "\" is not found for " << Design::getPinFullName(pin)
                << ".";
    const double configured_limit_ns = input.configured_max_sink_tran_ns;
    return configured_limit_ns > 0.0 ? configured_limit_ns : 0.0;
  }

  const auto port_name = normalizePortName(pin->get_name());
  auto* lib_port = lib_cell->get_cell_port_or_port_bus(port_name.c_str());
  if (lib_port != nullptr && lib_port->isInput() != 0U) {
    if (auto slew_limit_ns = lib_port->get_port_slew_limit(idb::AnalysisMode::kMax); slew_limit_ns.has_value() && *slew_limit_ns > 0.0) {
      return convertLibTimeToNs(lib_cell, *slew_limit_ns);
    }
  }

  auto* owner_lib = lib_cell->get_owner_lib();
  if (owner_lib != nullptr) {
    const auto default_max_transition = owner_lib->get_default_max_transition();
    const auto default_max_transition_ns = resolvePositiveMax({default_max_transition});
    if (default_max_transition_ns > 0.0) {
      return convertLibTimeToNs(lib_cell, default_max_transition_ns);
    }
  }

  const double configured_limit_ns = input.configured_max_sink_tran_ns;
  return configured_limit_ns > 0.0 ? configured_limit_ns : 0.0;
}

auto Wrapper::queryPinSlewLimit(const Config& config, const Pin* pin) const -> double
{
  return queryPinSlewLimit(PinSlewLimitInput{
      .pin = pin,
      .configured_max_sink_tran_ns = config.get_max_sink_tran(),
  });
}

auto Wrapper::queryCellHeightUm(const std::string& cell_master) const -> double
{
  if (_idb_layout == nullptr || _idb_layout->get_cell_master_list() == nullptr || _idb_layout->get_units() == nullptr) {
    LOG_WARNING << makeCharQueryContext("cell height", cell_master)
                << " failed: iDB layout metadata is not ready; auto-derived characterization unit may be unavailable.";
    return 0.0;
  }

  auto* idb_master = _idb_layout->get_cell_master_list()->find_cell_master(cell_master);
  if (idb_master == nullptr) {
    LOG_WARNING << makeCharQueryContext("cell height", cell_master)
                << " failed: iDB cell master is not found; auto-derived characterization unit may be unavailable.";
    return 0.0;
  }

  const int dbu_per_micron = _idb_layout->get_units()->get_micron_dbu();
  if (dbu_per_micron <= 0) {
    LOG_WARNING << makeCharQueryContext("cell height", cell_master)
                << " failed: invalid DBU-per-micron in iDB units; auto-derived characterization unit may be unavailable.";
    return 0.0;
  }

  return static_cast<double>(idb_master->get_height()) / static_cast<double>(dbu_per_micron);
}

auto Wrapper::queryCellAreaUm2(const std::string& cell_master) const -> double
{
  if (_idb_layout == nullptr || _idb_layout->get_cell_master_list() == nullptr || _idb_layout->get_units() == nullptr) {
    LOG_WARNING << makeCharQueryContext("cell area", cell_master) << " failed: iDB layout metadata is not ready.";
    return 0.0;
  }

  auto* idb_master = _idb_layout->get_cell_master_list()->find_cell_master(cell_master);
  if (idb_master == nullptr) {
    LOG_WARNING << makeCharQueryContext("cell area", cell_master) << " failed: iDB cell master is not found.";
    return 0.0;
  }

  const int dbu_per_micron = _idb_layout->get_units()->get_micron_dbu();
  if (dbu_per_micron <= 0) {
    LOG_WARNING << makeCharQueryContext("cell area", cell_master) << " failed: invalid DBU-per-micron in iDB units.";
    return 0.0;
  }

  const auto dbu_per_micron_double = static_cast<double>(dbu_per_micron);
  return (static_cast<double>(idb_master->get_width()) * static_cast<double>(idb_master->get_height()))
         / (dbu_per_micron_double * dbu_per_micron_double);
}

auto Wrapper::queryCharInputPinCap(const std::string& cell_master) const -> double
{
  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << makeCharQueryContext("input pin cap", cell_master) << " failed: liberty cell not found; return 0.0.";
    return 0.0;
  }
  idb::LibPort* input = nullptr;
  idb::LibPort* output = nullptr;
  lib_cell->bufferPorts(input, output);
  if (input == nullptr) {
    return 0.0;
  }
  return convertLibCapToPf(lib_cell, input->get_port_cap());
}

auto Wrapper::queryPinCapacitance(const Pin* pin) const -> double
{
  if (pin == nullptr) {
    LOG_WARNING << "Null pin provided when querying pin capacitance.";
    return 0.0;
  }

  auto* inst = pin->get_inst();
  const std::string pin_full_name = inst != nullptr ? (inst->get_name() + "/" + pin->get_name()) : pin->get_name();
  if (inst == nullptr) {
    LOG_WARNING << "Pin-cap query skipped: CTS pin has no owning instance for " << pin_full_name << ".";
    return 0.0;
  }

  const auto& cell_master = inst->get_cell_master();
  if (cell_master.empty()) {
    LOG_WARNING << "Pin-cap query skipped: CTS instance has no cell master for " << pin_full_name << ".";
    return 0.0;
  }

  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << "Pin-cap query skipped: liberty cell \"" << cell_master << "\" is not found for " << pin_full_name << ".";
    return 0.0;
  }

  const auto port_name = normalizePortName(pin->get_name());
  auto* lib_port = lib_cell->get_cell_port_or_port_bus(port_name.c_str());
  if (lib_port == nullptr) {
    LOG_WARNING << "Pin-cap query skipped: liberty port \"" << port_name << "\" is not found on cell " << cell_master << ".";
    return 0.0;
  }
  return queryLibPortCapacitancePf(lib_cell, lib_port);
}

auto Wrapper::queryRootDriverCostDirect(const std::string& cell_master, double input_slew_ns, double output_load_pf,
                                        double clock_period_ns) const -> RootDriverCost
{
  auto cost = makeInvalidCost("direct", cell_master, input_slew_ns, output_load_pf);
  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << "Direct root-driver cost query skipped: liberty cell is not found: " << cell_master << ".";
    return cost;
  }

  idb::LibPort* input_port = nullptr;
  idb::LibPort* output_port = nullptr;
  lib_cell->bufferPorts(input_port, output_port);
  if (input_port == nullptr || output_port == nullptr) {
    LOG_WARNING << "Direct root-driver cost query skipped: cell is not a single-input/single-output buffer or inverter: " << cell_master
                << ".";
    return cost;
  }

  const auto timing_arc_set = findBufferArcSet(lib_cell);
  auto power_arc_set = lib_cell->findLibertyPowerArcSet(input_port->get_port_name(), output_port->get_port_name());
  cost.cell_delay_ns = lookupRootCellDelayNs(lib_cell, timing_arc_set.value_or(nullptr), input_slew_ns, output_load_pf);
  cost.output_slew_ns = lookupRootCellOutputSlewNs(lib_cell, timing_arc_set.value_or(nullptr), input_slew_ns, output_load_pf);
  cost.internal_power_w = lookupInternalPowerW(lib_cell, power_arc_set.value_or(nullptr), input_slew_ns, output_load_pf, clock_period_ns);
  cost.leakage_power_w = lookupLeakagePowerW(lib_cell);
  cost.cell_power_w = cost.internal_power_w + cost.leakage_power_w;
  cost.valid = cost.cell_delay_ns > 0.0 || cost.output_slew_ns > 0.0 || cost.cell_power_w > 0.0;
  return cost;
}

auto Wrapper::queryBufferPorts(const std::string& cell_master) const -> std::pair<std::string, std::string>
{
  auto* lib_cell = findLibertyCell(cell_master);
  if (lib_cell == nullptr) {
    LOG_WARNING << makeCharQueryContext("buffer ports", cell_master) << " failed: liberty cell not found; return empty port names.";
    return {"", ""};
  }
  idb::LibPort* input = nullptr;
  idb::LibPort* output = nullptr;
  lib_cell->bufferPorts(input, output);
  const std::string in_name = input != nullptr ? input->get_port_name() : "";
  const std::string out_name = output != nullptr ? output->get_port_name() : "";
  return {in_name, out_name};
}

}  // namespace icts
