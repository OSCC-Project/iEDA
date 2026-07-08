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
 * @file CharacterizationRealTechFixture.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Compiled helpers for real-tech characterization tests.
 */

#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"

#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "Flow.hh"
#include "characterization/Characterization.hh"
#include "common/CTSTestRuntime.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/logging/ScopedLogFile.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "idm.h"
#include "io/Wrapper.hh"
#include "utils/logger/Schema.hh"

namespace icts_test::characterization::realtech {

namespace {

auto BuildRuntimeCharacterizationBufferCells(const std::vector<std::string>& buffer_types) -> std::vector<icts::CharacterizationBufferCell>
{
  std::vector<icts::CharacterizationBufferCell> buffer_cells;
  buffer_cells.reserve(buffer_types.size());
  for (const auto& cell_master : buffer_types) {
    auto [input_pin, output_pin] = icts_test::runtime::CurrentRuntime().wrapper.queryBufferPorts(cell_master);
    buffer_cells.push_back(icts::CharacterizationBufferCell{
        .cell_master = cell_master,
        .max_cap_pf = 0.0,
        .input_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCharInputPinCap(cell_master),
        .input_slew_limit_ns = icts_test::runtime::CurrentRuntime().wrapper.queryCellInPinSlewLimit(cell_master),
        .input_slew_table_axis_max_ns = icts_test::runtime::CurrentRuntime().wrapper.queryCellInPinSlewTableAxisMax(cell_master),
        .output_cap_limit_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapLimit(cell_master),
        .output_cap_table_axis_max_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapTableAxisMax(cell_master),
        .cell_height_um = icts_test::runtime::CurrentRuntime().wrapper.queryCellHeightUm(cell_master),
        .input_pin = std::move(input_pin),
        .output_pin = std::move(output_pin),
    });
  }
  return buffer_cells;
}

}  // namespace

auto CaptureConfigState() -> ConfigState
{
  ConfigState state{};
  state.skew_bound = icts_test::runtime::CurrentRuntime().config.get_skew_bound();
  state.max_buf_tran = icts_test::runtime::CurrentRuntime().config.get_max_buf_tran();
  state.root_input_slew = icts_test::runtime::CurrentRuntime().config.get_root_input_slew();
  state.max_sink_tran = icts_test::runtime::CurrentRuntime().config.get_max_sink_tran();
  state.max_cap = icts_test::runtime::CurrentRuntime().config.get_max_cap();
  state.has_max_buf_tran = icts_test::runtime::CurrentRuntime().config.has_max_buf_tran();
  state.has_max_cap = icts_test::runtime::CurrentRuntime().config.has_max_cap();
  state.max_length = icts_test::runtime::CurrentRuntime().config.get_max_length();
  state.wirelength_unit_um = icts_test::runtime::CurrentRuntime().config.get_wirelength_unit_um();
  state.wirelength_iterations = icts_test::runtime::CurrentRuntime().config.get_wirelength_iterations();
  state.slew_steps = icts_test::runtime::CurrentRuntime().config.get_slew_steps();
  state.cap_steps = icts_test::runtime::CurrentRuntime().config.get_cap_steps();
  state.wire_width = icts_test::runtime::CurrentRuntime().config.get_wire_width();
  state.max_fanout = icts_test::runtime::CurrentRuntime().config.get_max_fanout();
  state.routing_layers = icts_test::runtime::CurrentRuntime().config.get_routing_layers();
  state.buffer_types = icts_test::runtime::CurrentRuntime().config.get_buffer_types();
  state.char_buf_redundancy_pct = icts_test::runtime::CurrentRuntime().config.get_char_buf_redundancy_pct();
  state.force_branch_buffer = icts_test::runtime::CurrentRuntime().config.is_force_branch_buffer();
  state.htree_depth_explore_window = icts_test::runtime::CurrentRuntime().config.get_htree_depth_explore_window();
  state.enable_sink_clustering = icts_test::runtime::CurrentRuntime().config.is_enable_sink_clustering();
  state.work_dir = icts_test::runtime::CurrentRuntime().config.get_work_dir();
  state.log_file = icts_test::runtime::CurrentRuntime().config.get_log_file();
  state.visualization_dir = icts_test::runtime::CurrentRuntime().config.get_visualization_dir();
  state.statistics_dir = icts_test::runtime::CurrentRuntime().config.get_statistics_dir();
  return state;
}

auto ApplyConfigState(const ConfigState& state) -> void
{
  icts_test::runtime::CurrentRuntime().config.reset();
  icts_test::runtime::CurrentRuntime().config.set_skew_bound(state.skew_bound);
  if (state.has_max_buf_tran) {
    icts_test::runtime::CurrentRuntime().config.set_max_buf_tran(state.max_buf_tran);
  }
  icts_test::runtime::CurrentRuntime().config.set_root_input_slew(state.root_input_slew);
  icts_test::runtime::CurrentRuntime().config.set_max_sink_tran(state.max_sink_tran);
  if (state.has_max_cap) {
    icts_test::runtime::CurrentRuntime().config.set_max_cap(state.max_cap);
  }
  icts_test::runtime::CurrentRuntime().config.set_max_length(state.max_length);
  icts_test::runtime::CurrentRuntime().config.set_wirelength_unit_um(state.wirelength_unit_um);
  icts_test::runtime::CurrentRuntime().config.set_wirelength_iterations(state.wirelength_iterations);
  icts_test::runtime::CurrentRuntime().config.set_slew_steps(state.slew_steps);
  icts_test::runtime::CurrentRuntime().config.set_cap_steps(state.cap_steps);
  icts_test::runtime::CurrentRuntime().config.set_wire_width(state.wire_width);
  icts_test::runtime::CurrentRuntime().config.set_max_fanout(state.max_fanout);
  icts_test::runtime::CurrentRuntime().config.set_routing_layers(state.routing_layers);
  icts_test::runtime::CurrentRuntime().config.set_buffer_types(state.buffer_types);
  icts_test::runtime::CurrentRuntime().config.set_char_buf_redundancy_pct(state.char_buf_redundancy_pct);
  icts_test::runtime::CurrentRuntime().config.set_force_branch_buffer(state.force_branch_buffer);
  icts_test::runtime::CurrentRuntime().config.set_htree_depth_explore_window(state.htree_depth_explore_window);
  icts_test::runtime::CurrentRuntime().config.set_enable_sink_clustering(state.enable_sink_clustering);
  icts_test::runtime::CurrentRuntime().config.set_work_dir(state.work_dir);
  icts_test::runtime::CurrentRuntime().config.set_log_file(state.log_file);
  icts_test::runtime::CurrentRuntime().config.set_visualization_dir(state.visualization_dir);
  icts_test::runtime::CurrentRuntime().config.set_statistics_dir(state.statistics_dir);
}

auto MakeRuntimeCharBuilderContract() -> RuntimeCharBuilderContract
{
  RuntimeCharBuilderContract contract;
  if (icts_test::runtime::CurrentRuntime().config.has_max_buf_tran()
      && icts_test::runtime::CurrentRuntime().config.get_max_buf_tran() > 0.0) {
    contract.config.max_slew_ns = icts_test::runtime::CurrentRuntime().config.get_max_buf_tran();
  }
  if (icts_test::runtime::CurrentRuntime().config.has_max_cap() && icts_test::runtime::CurrentRuntime().config.get_max_cap() > 0.0) {
    contract.config.max_cap_pf = icts_test::runtime::CurrentRuntime().config.get_max_cap();
  }
  if (icts_test::runtime::CurrentRuntime().config.get_wirelength_unit_um() > 0.0) {
    contract.config.wirelength_unit_um = icts_test::runtime::CurrentRuntime().config.get_wirelength_unit_um();
  }
  contract.config.wirelength_iterations = icts_test::runtime::CurrentRuntime().config.get_wirelength_iterations();
  contract.config.slew_steps = icts_test::runtime::CurrentRuntime().config.get_slew_steps();
  contract.config.cap_steps = icts_test::runtime::CurrentRuntime().config.get_cap_steps();
  contract.input.buffer_types = icts_test::runtime::CurrentRuntime().config.get_buffer_types();
  contract.input.characterization_buffer_cells = BuildRuntimeCharacterizationBufferCells(contract.input.buffer_types);
  contract.config.char_buf_redundancy_pct = icts_test::runtime::CurrentRuntime().config.get_char_buf_redundancy_pct();

  const auto& routing_layers = icts_test::runtime::CurrentRuntime().config.get_routing_layers();
  if (!routing_layers.empty()) {
    contract.config.routing_layer = static_cast<int>(routing_layers.front());
  }
  if (icts_test::runtime::CurrentRuntime().config.get_wire_width() > 0.0) {
    contract.config.wire_width_um = icts_test::runtime::CurrentRuntime().config.get_wire_width();
  }
  contract.input.clock_route_segment_rc
      = icts_test::runtime::CurrentRuntime().wrapper.queryConfiguredClockRouteSegmentRc(icts_test::runtime::CurrentRuntime().config);
  const auto dbu_per_um = icts_test::runtime::CurrentRuntime().wrapper.queryDbUnit();
  if (dbu_per_um > 0) {
    contract.input.dbu_per_um = dbu_per_um;
  }
  contract.input.root_input_slew_ns = std::max(0.0, icts_test::runtime::CurrentRuntime().config.get_root_input_slew());
  contract.input.wrapper = &icts_test::runtime::CurrentRuntime().wrapper;
  contract.input.fast_sta = &icts_test::runtime::CurrentRuntime().fast_sta;
  contract.input.reporter = &icts_test::runtime::CurrentRuntime().reporter;
  return contract;
}

auto MakeRealTechCharConfigState(const ConfigState& baseline_state, std::optional<std::vector<std::string>> buffer_types,
                                 double max_buf_tran_ns, double max_cap_pf, bool omit_wirelength_unit, bool force_branch_buffer)
    -> ConfigState
{
  auto configured_state = baseline_state;
  configured_state.has_max_buf_tran = max_buf_tran_ns > 0.0;
  configured_state.max_buf_tran = max_buf_tran_ns;
  configured_state.has_max_cap = max_cap_pf > 0.0;
  configured_state.max_cap = max_cap_pf;
  configured_state.wirelength_unit_um = omit_wirelength_unit ? 0.0 : kRealTechCharWirelengthUnitUm;
  configured_state.wirelength_iterations = kRealTechCharWirelengthIterations;
  configured_state.slew_steps = kRealTechCharSlewSteps;
  configured_state.cap_steps = kRealTechCharCapSteps;
  configured_state.char_buf_redundancy_pct = 0.0;
  configured_state.force_branch_buffer = force_branch_buffer;
  if (buffer_types.has_value()) {
    configured_state.buffer_types = *buffer_types;
  }
  return configured_state;
}

RealTechCharFixture::RealTechCharFixture() = default;

RealTechCharFixture::~RealTechCharFixture()
{
  restore();
}

auto RealTechCharFixture::prepare(const std::string& scenario_name, std::optional<std::vector<std::string>> buffer_types,
                                  double max_buf_tran_ns, double max_cap_pf, bool omit_wirelength_unit, bool force_branch_buffer)
    -> std::optional<std::string>
{
  if (_is_prepared) {
    restore();
  }

  const auto& setup_state = icts_test::common::realtech::EnsureRealTechSetup();
  if (setup_state.mode != icts_test::common::realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    return setup_state.summary;
  }

  const auto& cts_config_path = setup_state.cts_config_path;
  if (cts_config_path.empty() || !std::filesystem::exists(cts_config_path)) {
    return "Cannot resolve real-tech CTS config path from setup state.";
  }

  const auto original_config_state = CaptureConfigState();
  auto configured_state = MakeRealTechCharConfigState(original_config_state, std::move(buffer_types), max_buf_tran_ns, max_cap_pf,
                                                      omit_wirelength_unit, force_branch_buffer);
  _original_config_state = original_config_state;
  ApplyConfigState(configured_state);

  const auto output_dir = icts_test::common::io::ResolveOutputDir() / "characterization" / "realtech" / scenario_name;
  std::error_code error_code;
  std::filesystem::create_directories(output_dir, error_code);
  if (error_code) {
    return "Cannot create real-tech characterization output directory.";
  }

  _cts_log_guard = std::make_unique<icts_test::common::logging::ScopedLogFile>(output_dir / "cts.log", "Characterization Test Report");
  icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable("Characterization Scenario",
                                                                  {
                                                                      {"scenario", scenario_name},
                                                                      {"omit_wirelength_unit", omit_wirelength_unit ? "true" : "false"},
                                                                      {"force_branch_buffer", force_branch_buffer ? "true" : "false"},
                                                                  });
  _is_prepared = true;
  return std::nullopt;
}

auto RealTechCharFixture::restore() -> void
{
  if (!_is_prepared || !_original_config_state.has_value()) {
    return;
  }

  ApplyConfigState(*_original_config_state);
  icts_test::runtime::CurrentRuntime().wrapper.reset();
  if (auto* idb_builder = dmInst->get_idb_builder(); idb_builder != nullptr) {
    icts_test::runtime::CurrentRuntime().wrapper.init(idb_builder);
  }
  _is_prepared = false;
  _original_config_state.reset();
  _cts_log_guard.reset();
}

auto JoinStrings(const std::vector<std::string>& values) -> std::string
{
  std::ostringstream output_stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      output_stream << ",";
    }
    output_stream << values.at(index);
  }
  return output_stream.str();
}

auto WriteScenarioLog(const std::string& scenario_name, const std::string& file_name, const std::string& content) -> bool
{
  const auto output_dir = icts_test::common::io::ResolveOutputDir() / "characterization" / "realtech" / scenario_name;
  std::error_code error_code;
  std::filesystem::create_directories(output_dir, error_code);
  if (error_code) {
    return false;
  }
  return icts_test::common::io::WriteTextLog(output_dir / file_name, content);
}

auto CollectConfiguredBufferLimitInfo() -> std::vector<BufferLimitInfo>
{
  std::vector<BufferLimitInfo> infos;
  std::set<std::string> seen_cell_masters;

  for (const auto& cell_master : icts_test::runtime::CurrentRuntime().config.get_buffer_types()) {
    if (!seen_cell_masters.insert(cell_master).second) {
      continue;
    }

    auto [input_pin, output_pin] = icts_test::runtime::CurrentRuntime().wrapper.queryBufferPorts(cell_master);
    if (input_pin.empty() || output_pin.empty()) {
      continue;
    }

    infos.push_back(BufferLimitInfo{
        .cell_master = cell_master,
        .input_pin = std::move(input_pin),
        .output_pin = std::move(output_pin),
        .port_slew_limit_ns = icts_test::runtime::CurrentRuntime().wrapper.queryCellInPinSlewLimit(cell_master),
        .table_slew_limit_ns = icts_test::runtime::CurrentRuntime().wrapper.queryCellInPinSlewTableAxisMax(cell_master),
        .port_cap_limit_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapLimit(cell_master),
        .table_cap_limit_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapTableAxisMax(cell_master),
    });
  }

  return infos;
}

auto CollectUsableBufferMasters(const std::vector<BufferLimitInfo>& infos) -> std::vector<std::string>
{
  std::vector<std::string> masters;
  for (const auto& info : infos) {
    const bool has_slew_limit = info.port_slew_limit_ns > 0.0 || info.table_slew_limit_ns > 0.0;
    const bool has_cap_limit = info.port_cap_limit_pf > 0.0 || info.table_cap_limit_pf > 0.0;
    if (has_slew_limit && has_cap_limit) {
      masters.push_back(info.cell_master);
    }
  }
  return masters;
}

auto LookupBufferInfo(const std::vector<BufferLimitInfo>& infos, const std::string& cell_master) -> const BufferLimitInfo*
{
  auto it = std::ranges::find_if(infos, [&cell_master](const BufferLimitInfo& info) -> bool { return info.cell_master == cell_master; });
  return it == infos.end() ? nullptr : &(*it);
}

auto MinPositiveResolvedLimit(const std::vector<BufferLimitInfo>& infos, const std::vector<std::string>& selected_masters, bool for_slew)
    -> double
{
  double port_min = std::numeric_limits<double>::infinity();
  double table_min = std::numeric_limits<double>::infinity();

  for (const auto& cell_master : selected_masters) {
    const auto* info = LookupBufferInfo(infos, cell_master);
    if (info == nullptr) {
      continue;
    }

    const double port_value = for_slew ? info->port_slew_limit_ns : info->port_cap_limit_pf;
    const double table_value = for_slew ? info->table_slew_limit_ns : info->table_cap_limit_pf;
    if (port_value > 0.0) {
      port_min = std::min(port_min, port_value);
    }
    if (table_value > 0.0) {
      table_min = std::min(table_min, table_value);
    }
  }

  if (std::isfinite(port_min)) {
    return port_min;
  }
  if (std::isfinite(table_min)) {
    return table_min;
  }
  return 0.0;
}

auto ResolveDefaultWirelengthUnitUm(const std::vector<BufferLimitInfo>& infos, const std::vector<std::string>& selected_masters) -> double
{
  double strongest_drive_cap_pf = -1.0;
  double resolved_unit_um = 0.0;

  for (const auto& cell_master : selected_masters) {
    const auto* info = LookupBufferInfo(infos, cell_master);
    if (info == nullptr) {
      continue;
    }

    const double drive_cap_pf = info->port_cap_limit_pf > 0.0 ? info->port_cap_limit_pf : info->table_cap_limit_pf;
    if (drive_cap_pf <= 0.0) {
      continue;
    }

    const double cell_height_um = icts_test::runtime::CurrentRuntime().wrapper.queryCellHeightUm(cell_master);
    if (cell_height_um <= 0.0) {
      continue;
    }

    if (drive_cap_pf > strongest_drive_cap_pf) {
      strongest_drive_cap_pf = drive_cap_pf;
      resolved_unit_um = cell_height_um * 10.0;
    }
  }

  return resolved_unit_um;
}

}  // namespace icts_test::characterization::realtech
