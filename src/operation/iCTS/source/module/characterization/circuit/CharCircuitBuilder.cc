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
 * @file CharCircuitBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Temporary characterization circuit and parasitic setup.
 */

#include "characterization/circuit/CharCircuitBuilder.hh"

#include <glog/logging.h>

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "characterization/builder/CharFeasibilityChecker.hh"

namespace icts::char_builder::detail {

auto CharCircuitBuilder::createCharCircuit(const TopologyDesc& topo, const std::vector<std::string>& buf_masters) -> void
{
  _impl._temp_inst_names.clear();
  _impl._temp_net_names.clear();
  _impl._fast_sta_char_context_id = ::icts::kInvalidFastStaCharContextId;

  const std::string id_prefix = "cts_char_" + std::to_string(_impl._char_circuit_id) + "_";

  const auto& source_buf = _impl._sorted_buffers.back();
  const auto& sink_buf = _impl._sorted_buffers.front();

  _impl._source_inst_name = id_prefix + "source";
  _impl._source_in_pin = _impl._source_inst_name + "/" + source_buf.input_pin;
  _impl._source_out_pin = _impl._source_inst_name + "/" + source_buf.output_pin;
  _impl._timing_observation_pin.clear();

  _impl._sink_inst_name = id_prefix + "sink";
  _impl._sink_in_pin = _impl._sink_inst_name + "/" + sink_buf.input_pin;

  for (size_t i = 0; i < buf_masters.size(); ++i) {
    const std::string inst_name = id_prefix + "buf_" + std::to_string(i);
    _impl._temp_inst_names.push_back(inst_name);
  }

  for (size_t i = 0; i < topo.wire_segments_um.size(); ++i) {
    const std::string net_name = id_prefix + "net_" + std::to_string(i);
    _impl._temp_net_names.push_back(net_name);
  }

  for (size_t bi = 0; bi < buf_masters.size(); ++bi) {
    const ::icts::CharacterizationBufferCell* buffer_cell_ptr
        = _impl.feasibilityChecker().findCharacterizationBufferCell(buf_masters.at(bi));
    if (buffer_cell_ptr == nullptr) {
      LOG_FATAL << "Characterization buffer cell not found for: " << buf_masters.at(bi);
      return;
    }
    const auto& buffer_cell = *buffer_cell_ptr;

    if (bi + 1U == buf_masters.size()) {
      _impl._timing_observation_pin = _impl._temp_inst_names.at(bi) + "/" + buffer_cell.output_pin;
    }
  }

  if (_impl._timing_observation_pin.empty()) {
    _impl._timing_observation_pin = _impl._sink_in_pin;
  }

  _impl._char_clock_name = id_prefix + "clk";
  LOG_FATAL_IF(_impl._wrapper == nullptr) << "CharCircuitBuilder: Wrapper dependency is not configured.";
  LOG_FATAL_IF(_impl._fast_sta == nullptr) << "CharCircuitBuilder: FastSTA dependency is not configured.";
  _impl._fast_sta_char_context_id = _impl._fast_sta->buildCharContext(::icts::FastStaCharTopologySpec{
      .wrapper = _impl._wrapper,
      .source_cell_master = source_buf.cell_master,
      .sink_cell_master = sink_buf.cell_master,
      .buffer_cell_masters = buf_masters,
      .wire_segments_um = topo.wire_segments_um,
      .dbu_per_um = _impl._dbu_per_um,
      .routing_layer = _impl._routing_layer,
      .wire_width_um = _impl._wire_width_um,
      .clock_period_ns = 10.0,
      .root_input_slew_ns = _impl._root_input_slew_ns,
  });

  ++_impl._char_circuit_id;
}

auto CharCircuitBuilder::setCharParasitics(const TopologyDesc& topo, double load_pf) const -> void
{
  (void) topo;
  LOG_FATAL_IF(_impl._fast_sta_char_context_id == ::icts::kInvalidFastStaCharContextId)
      << "Fast STA characterization context is not prepared before parasitic load update.";
  LOG_FATAL_IF(_impl._fast_sta == nullptr) << "CharCircuitBuilder: FastSTA dependency is not configured.";
  (void) _impl._fast_sta->setCharLoad(_impl._fast_sta_char_context_id, load_pf);
}

auto CharCircuitBuilder::destroyCharCircuit() -> void
{
  if (_impl._fast_sta_char_context_id != ::icts::kInvalidFastStaCharContextId) {
    LOG_FATAL_IF(_impl._fast_sta == nullptr) << "CharCircuitBuilder: FastSTA dependency is not configured.";
    (void) _impl._fast_sta->eraseCharContext(_impl._fast_sta_char_context_id);
    _impl._fast_sta_char_context_id = ::icts::kInvalidFastStaCharContextId;
  }
  _impl._sink_inst_name.clear();
  _impl._source_inst_name.clear();
  _impl._temp_net_names.clear();
  _impl._temp_inst_names.clear();
  _impl._source_in_pin.clear();
  _impl._source_out_pin.clear();
  _impl._sink_in_pin.clear();
  _impl._timing_observation_pin.clear();
}

}  // namespace icts::char_builder::detail
