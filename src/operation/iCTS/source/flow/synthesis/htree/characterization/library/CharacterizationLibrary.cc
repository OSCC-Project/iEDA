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
 * @file CharacterizationLibrary.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Reusable characterization cache shared by CTS synthesis stages.
 */

#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"

#include <glog/logging.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "characterization/Characterization.hh"
#include "config/Config.hh"
#include "io/Wrapper.hh"

namespace icts {
namespace {

auto BuildRuntimeCharacterizationBufferCells(Wrapper& wrapper, const std::vector<std::string>& buffer_types)
    -> std::vector<CharacterizationBufferCell>
{
  std::vector<CharacterizationBufferCell> buffer_cells;
  buffer_cells.reserve(buffer_types.size());
  for (const auto& cell_master : buffer_types) {
    auto [input_pin, output_pin] = wrapper.queryBufferPorts(cell_master);
    buffer_cells.push_back(CharacterizationBufferCell{
        .cell_master = cell_master,
        .max_cap_pf = 0.0,
        .input_cap_pf = wrapper.queryCharInputPinCap(cell_master),
        .input_slew_limit_ns = wrapper.queryCellInPinSlewLimit(cell_master),
        .input_slew_table_axis_max_ns = wrapper.queryCellInPinSlewTableAxisMax(cell_master),
        .output_cap_limit_pf = wrapper.queryCellOutPinCapLimit(cell_master),
        .output_cap_table_axis_max_pf = wrapper.queryCellOutPinCapTableAxisMax(cell_master),
        .cell_height_um = wrapper.queryCellHeightUm(cell_master),
        .input_pin = std::move(input_pin),
        .output_pin = std::move(output_pin),
    });
  }
  return buffer_cells;
}

}  // namespace

auto CharacterizationLibrary::makeCharacterizationCacheKey(const CharBuilder::Input& input, const CharBuilder::Config& config)
    -> CharacterizationCacheKey
{
  return CharacterizationCacheKey{
      .wirelength_unit_um = config.wirelength_unit_um,
      .wirelength_iterations = config.wirelength_iterations,
      .wirelength_indices = config.wirelength_indices,
      .max_slew_ns = config.max_slew_ns,
      .max_cap_pf = config.max_cap_pf,
      .buffer_types = input.buffer_types,
      .characterization_buffer_cells = input.characterization_buffer_cells,
      .char_buf_redundancy_pct = config.char_buf_redundancy_pct,
      .slew_steps = config.slew_steps,
      .cap_steps = config.cap_steps,
      .routing_layer = config.routing_layer,
      .wire_width_um = config.wire_width_um,
      .clock_route_segment_rc = input.clock_route_segment_rc,
      .dbu_per_um = input.dbu_per_um,
      .root_input_slew_ns = input.root_input_slew_ns,
  };
}

auto CharacterizationLibrary::ensure(const CharBuilder::Input& input, const CharBuilder::Config& config) -> EnsureSummary
{
  const auto characterization_cache_key = makeCharacterizationCacheKey(input, config);
  if (_ready && characterization_cache_key == _characterization_cache_key && !_char_builder.get_segment_chars().empty()) {
    return EnsureSummary{.success = true, .reused = true, .failure_reason = ""};
  }

  _char_builder.init(input, config);
  _char_builder.build();
  if (_char_builder.get_segment_chars().empty() || _char_builder.get_wirelength_unit_um() <= 0.0) {
    LOG_WARNING << "CharacterizationLibrary: characterization did not produce reusable segment chars.";
    _ready = false;
    _characterization_cache_key = {};
    return EnsureSummary{.success = false, .reused = false, .failure_reason = "no_reusable_segment_chars"};
  }

  _characterization_cache_key = characterization_cache_key;
  _ready = true;
  return EnsureSummary{.success = true, .reused = false, .failure_reason = ""};
}

auto CharacterizationLibrary::buildRuntimeInput(const CharacterizationRuntimeInput& runtime_input) -> CharBuilder::Input
{
  LOG_FATAL_IF(runtime_input.config == nullptr) << "CharacterizationLibrary: runtime config is null.";
  LOG_FATAL_IF(runtime_input.wrapper == nullptr) << "CharacterizationLibrary: runtime wrapper is null.";
  LOG_FATAL_IF(runtime_input.fast_sta == nullptr) << "CharacterizationLibrary: runtime FastSTA is null.";
  LOG_FATAL_IF(runtime_input.reporter == nullptr) << "CharacterizationLibrary: runtime reporter is null.";

  const auto& config = *runtime_input.config;
  auto& wrapper = *runtime_input.wrapper;
  CharBuilder::Input char_input;
  char_input.buffer_types = config.get_buffer_types();
  char_input.characterization_buffer_cells = BuildRuntimeCharacterizationBufferCells(wrapper, char_input.buffer_types);
  char_input.clock_route_segment_rc = wrapper.queryConfiguredClockRouteSegmentRc(config);
  const auto dbu_per_um = wrapper.queryDbUnit();
  LOG_FATAL_IF(dbu_per_um <= 0) << "CharacterizationLibrary: DBU-per-micron must be available before characterization.";
  char_input.dbu_per_um = dbu_per_um;
  char_input.root_input_slew_ns = std::max(0.0, config.get_root_input_slew());
  char_input.wrapper = &wrapper;
  char_input.fast_sta = runtime_input.fast_sta;
  char_input.reporter = runtime_input.reporter;
  return char_input;
}

auto CharacterizationLibrary::buildRuntimeConfig(const Config& config) -> CharBuilder::Config
{
  CharBuilder::Config char_config;
  if (config.has_max_buf_tran() && config.get_max_buf_tran() > 0.0) {
    char_config.max_slew_ns = config.get_max_buf_tran();
  }
  if (config.has_max_cap() && config.get_max_cap() > 0.0) {
    char_config.max_cap_pf = config.get_max_cap();
  }
  if (config.get_wirelength_unit_um() > 0.0) {
    char_config.wirelength_unit_um = config.get_wirelength_unit_um();
  }
  char_config.wirelength_iterations = config.get_wirelength_iterations();
  char_config.slew_steps = config.get_slew_steps();
  char_config.cap_steps = config.get_cap_steps();
  char_config.char_buf_redundancy_pct = config.get_char_buf_redundancy_pct();

  const auto& routing_layers = config.get_routing_layers();
  LOG_FATAL_IF(routing_layers.empty() || routing_layers.front() == 0U)
      << "CharacterizationLibrary: routing layer must be configured before characterization.";
  char_config.routing_layer = static_cast<int>(routing_layers.front());
  if (config.get_wire_width() > 0.0) {
    char_config.wire_width_um = config.get_wire_width();
  }
  return char_config;
}

}  // namespace icts
