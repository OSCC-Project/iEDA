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
 * @file FastStaPower.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS fast STA power calculation implementation.
 */

#include "FastStaPower.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "FastStaClockState.hh"
#include "FastStaLibertyModel.hh"
#include "clock_sizing/FastStaClockSizingEdit.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {
namespace {

auto lookupPowerTable(const std::vector<FastStaLibertyTable>& tables, double input_slew_ns, double output_load_pf) -> double
{
  for (const auto& table : tables) {
    if (!table.empty()) {
      return std::max(0.0, table.lookup(input_slew_ns, output_load_pf).value_or(0.0));
    }
  }
  return 0.0;
}

auto clockActivityDensity(double clock_period_ns) -> double
{
  return clock_period_ns > 0.0 ? 2.0 / clock_period_ns : 0.0;
}

auto resolveVoltage(const FastStaClockContext& context) -> double
{
  for (const auto& [_, cell] : context.liberty_cell_by_master) {
    if (cell.voltage_v > 0.0) {
      return cell.voltage_v;
    }
  }
  return 0.0;
}

auto findBufferInputNode(const FastStaClockContext& context, const FastStaNode& output_node) -> FastStaNodeId
{
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    const auto& node = context.nodes.at(node_id);
    if (node.kind == FastStaNodeKind::kBufferInput && node.inst_name == output_node.inst_name) {
      return node_id;
    }
  }
  return kInvalidFastStaNodeId;
}

auto calcNetSwitchingPowerW(const FastStaNet& net, double voltage, double activity_density) -> double
{
  return 0.5 * std::max(0.0, net.load_cap_pf) * 1e-12 * voltage * voltage * activity_density * 1e9;
}

auto calcBufferPower(FastStaClockContext& context, FastStaNodeId output_node_id, double activity_density) -> void
{
  if (output_node_id >= context.nodes.size()) {
    return;
  }
  auto& node = context.nodes.at(output_node_id);
  node.area_um2 = 0.0;
  node.leakage_power_w = 0.0;
  node.internal_power_w = 0.0;
  if (node.kind != FastStaNodeKind::kBufferOutput) {
    return;
  }
  const auto cell_iter = context.liberty_cell_by_master.find(node.cell_master);
  if (cell_iter == context.liberty_cell_by_master.end()) {
    return;
  }
  node.area_um2 = std::max(0.0, cell_iter->second.area_um2);
  node.leakage_power_w = std::max(0.0, cell_iter->second.leakage_power_w);
  if (!node.output_net_ids.empty() && node.output_net_ids.front() < context.nets.size()) {
    const auto& net = context.nets.at(node.output_net_ids.front());
    const auto input_node_id = findBufferInputNode(context, node);
    const auto input_slew_ns = input_node_id < context.nodes.size() ? context.nodes.at(input_node_id).timing.slew_ns : 0.0;
    const auto internal_energy_mw_ns = lookupPowerTable(cell_iter->second.timing_arc.internal_power_tables, input_slew_ns, net.load_cap_pf);
    node.internal_power_w = internal_energy_mw_ns * 1e-12 * activity_density * 1e9;
  }
}

auto sumPower(const FastStaClockContext& context) -> FastStaPowerSummary
{
  FastStaPowerSummary power;
  for (const auto& net : context.nets) {
    power.switching_power_w += net.switching_power_w;
  }
  std::unordered_set<std::string> seen_buffer_insts;
  seen_buffer_insts.reserve(context.nodes.size());
  for (const auto& node : context.nodes) {
    if (node.kind != FastStaNodeKind::kBufferOutput || node.inst_name.empty() || seen_buffer_insts.contains(node.inst_name)) {
      continue;
    }
    seen_buffer_insts.insert(node.inst_name);
    power.area_um2 += node.area_um2;
    power.leakage_power_w += node.leakage_power_w;
    power.internal_power_w += node.internal_power_w;
  }
  power.total_power_w = power.switching_power_w + power.internal_power_w + power.leakage_power_w;
  return power;
}

}  // namespace

auto FastStaPower::update(FastStaClockContext& context) -> bool
{
  const auto activity_density = clockActivityDensity(context.clock_period_ns);
  const auto voltage = resolveVoltage(context);
  for (auto& net : context.nets) {
    net.switching_power_w = calcNetSwitchingPowerW(net, voltage, activity_density);
  }
  for (FastStaNodeId node_id = 0U; node_id < context.nodes.size(); ++node_id) {
    calcBufferPower(context, node_id, activity_density);
  }
  context.power = sumPower(context);
  context.power_valid = true;
  return true;
}

auto FastStaPower::updateRegion(FastStaClockContext& context, const FastStaDirtyRegion& dirty_region) -> bool
{
  if (!dirty_region.valid) {
    return false;
  }
  if (!context.power_valid) {
    return update(context);
  }
  const auto activity_density = clockActivityDensity(context.clock_period_ns);
  const auto voltage = resolveVoltage(context);
  for (const auto net_id : dirty_region.net_ids) {
    if (net_id < context.nets.size()) {
      context.nets.at(net_id).switching_power_w = calcNetSwitchingPowerW(context.nets.at(net_id), voltage, activity_density);
    }
  }
  for (const auto node_id : dirty_region.node_ids) {
    if (node_id < context.nodes.size() && context.nodes.at(node_id).kind == FastStaNodeKind::kBufferOutput) {
      calcBufferPower(context, node_id, activity_density);
    }
  }
  context.power = sumPower(context);
  context.power_valid = true;
  return true;
}

}  // namespace icts
