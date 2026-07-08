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
 * @file FastStaClockState.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS fast STA per-clock graph state.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "liberty/FastStaLibertyModel.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {

enum class FastStaNodeKind
{
  kSource,
  kBufferInput,
  kBufferOutput,
  kSink
};

struct FastStaNode
{
  FastStaNodeKind kind = FastStaNodeKind::kSink;
  std::string name;
  std::string inst_name;
  std::string pin_name;
  std::string cell_master;
  FastStaPoint location;
  double input_cap_pf = 0.0;
  double max_slew_ns = 0.0;
  FastStaNetId incoming_net_id = kInvalidFastStaNetId;
  std::vector<FastStaNetId> output_net_ids;
  FastStaTimingPoint timing;
  double internal_power_w = 0.0;
  double leakage_power_w = 0.0;
  double area_um2 = 0.0;
};

struct FastStaNet
{
  std::string name;
  FastStaNodeId driver_node_id = kInvalidFastStaNodeId;
  std::vector<FastStaNodeId> load_node_ids;
  double wire_resistance_ohm = 0.0;
  double wire_cap_pf = 0.0;
  double load_cap_pf = 0.0;
  double max_cap_pf = 0.0;
  FastStaNetParasitic parasitic;
  FastStaDmpDriverResult driver_dmp;
  double switching_power_w = 0.0;
};

struct FastStaClockContext
{
  Wrapper* wrapper = nullptr;
  std::string clock_name;
  std::string clock_net_name;
  double clock_period_ns = 0.0;
  double root_input_slew_ns = 0.0;
  int32_t dbu_per_um = 0;
  int routing_layer = 0;
  std::optional<double> wire_width_um;
  FastStaNodeId source_node_id = kInvalidFastStaNodeId;
  std::vector<FastStaNode> nodes;
  std::vector<FastStaNet> nets;
  std::unordered_map<std::string, FastStaNodeId> node_id_by_name;
  std::unordered_map<std::pair<int, int>, FastStaNodeId, FastStaPointKeyHash> node_id_by_location;
  std::unordered_map<std::string, FastStaNetId> net_id_by_name;
  std::unordered_map<std::string, FastStaLibertyCell> liberty_cell_by_master;
  FastStaSkewSummary skew;
  FastStaPowerSummary power;
  bool timing_valid = false;
  bool power_valid = false;
};

}  // namespace icts
