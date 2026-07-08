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
 * @file FastStaIncremental.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Incremental update coordinator implementation for CTS fast STA contexts.
 */

#include "FastStaIncremental.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FastStaClockState.hh"
#include "FastStaLiberty.hh"
#include "FastStaLibertyModel.hh"
#include "Log.hh"
#include "clock_sizing/FastStaClockSizingEdit.hh"

namespace icts {
namespace {

auto normalizeBufferInputNodeId(const FastStaClockContext& context, FastStaNodeId node_id) -> FastStaNodeId
{
  if (node_id >= context.nodes.size()) {
    return kInvalidFastStaNodeId;
  }
  const auto& node = context.nodes.at(node_id);
  if (node.kind == FastStaNodeKind::kBufferInput) {
    return node_id;
  }
  if (node.kind != FastStaNodeKind::kBufferOutput || node.inst_name.empty()) {
    return kInvalidFastStaNodeId;
  }
  for (FastStaNodeId candidate_id = 0U; candidate_id < context.nodes.size(); ++candidate_id) {
    const auto& candidate = context.nodes.at(candidate_id);
    if (candidate.kind == FastStaNodeKind::kBufferInput && candidate.inst_name == node.inst_name) {
      return candidate_id;
    }
  }
  return kInvalidFastStaNodeId;
}

auto markReachableFromNode(const FastStaClockContext& context, FastStaNodeId node_id, FastStaDirtyRegion& dirty_region,
                           std::unordered_set<FastStaNodeId>& node_seen, std::unordered_set<FastStaNetId>& net_seen) -> void
{
  std::vector<FastStaNodeId> pending_nodes{node_id};
  while (!pending_nodes.empty()) {
    const auto current_node_id = pending_nodes.back();
    pending_nodes.pop_back();
    if (current_node_id >= context.nodes.size() || node_seen.contains(current_node_id)) {
      continue;
    }

    node_seen.insert(current_node_id);
    dirty_region.node_ids.push_back(current_node_id);
    const auto& node = context.nodes.at(current_node_id);
    if (node.kind == FastStaNodeKind::kBufferInput) {
      for (FastStaNodeId output_id = 0U; output_id < context.nodes.size(); ++output_id) {
        const auto& output_node = context.nodes.at(output_id);
        if (output_node.kind == FastStaNodeKind::kBufferOutput && output_node.inst_name == node.inst_name) {
          pending_nodes.push_back(output_id);
          break;
        }
      }
      continue;
    }

    for (const auto net_id : node.output_net_ids) {
      if (net_id >= context.nets.size()) {
        continue;
      }
      if (!net_seen.contains(net_id)) {
        net_seen.insert(net_id);
        dirty_region.net_ids.push_back(net_id);
      }
      for (const auto load_node_id : context.nets.at(net_id).load_node_ids) {
        pending_nodes.push_back(load_node_id);
      }
    }
  }
}

auto collectDirtyRegion(const FastStaClockContext& context, FastStaNodeId changed_input_node_id) -> FastStaDirtyRegion
{
  FastStaDirtyRegion dirty_region;
  if (changed_input_node_id >= context.nodes.size()) {
    return dirty_region;
  }

  auto start_node_id = changed_input_node_id;
  const auto incoming_net_id = context.nodes.at(changed_input_node_id).incoming_net_id;
  if (incoming_net_id < context.nets.size()) {
    const auto incoming_driver_id = context.nets.at(incoming_net_id).driver_node_id;
    if (incoming_driver_id < context.nodes.size() && context.nodes.at(incoming_driver_id).kind == FastStaNodeKind::kBufferOutput) {
      const auto driver_input_id = normalizeBufferInputNodeId(context, incoming_driver_id);
      if (driver_input_id != kInvalidFastStaNodeId) {
        start_node_id = driver_input_id;
      }
    } else if (incoming_driver_id < context.nodes.size()) {
      start_node_id = incoming_driver_id;
    }
  }

  dirty_region.valid = true;
  dirty_region.start_node_id = start_node_id;
  std::unordered_set<FastStaNodeId> node_seen;
  std::unordered_set<FastStaNetId> net_seen;
  markReachableFromNode(context, start_node_id, dirty_region, node_seen, net_seen);
  return dirty_region;
}

auto applyBufferMasterChange(FastStaClockContext& context, FastStaNodeId node_id, std::string_view cell_master, bool invalidate_context)
    -> FastStaNodeId
{
  if (node_id >= context.nodes.size()) {
    LOG_ERROR << "FastStaIncremental: buffer master change skipped because node id is invalid.";
    return kInvalidFastStaNodeId;
  }
  auto& node = context.nodes.at(node_id);
  if (node.kind != FastStaNodeKind::kBufferInput && node.kind != FastStaNodeKind::kBufferOutput) {
    LOG_ERROR << "FastStaIncremental: node \"" << node.name << "\" is not a buffer node.";
    return kInvalidFastStaNodeId;
  }
  const auto input_node_id = normalizeBufferInputNodeId(context, node_id);
  if (input_node_id == kInvalidFastStaNodeId) {
    LOG_ERROR << "FastStaIncremental: buffer master change skipped because buffer input node is unavailable for \"" << node.name << "\".";
    return kInvalidFastStaNodeId;
  }
  const auto target_master = std::string(cell_master);
  if (!context.liberty_cell_by_master.contains(target_master)) {
    LOG_FATAL_IF(context.wrapper == nullptr) << "FastStaIncremental: Wrapper is unavailable.";
    context.liberty_cell_by_master[target_master] = FastStaLiberty::extractBufferCell(*context.wrapper, target_master);
  }
  const auto inst_name = context.nodes.at(input_node_id).inst_name;
  for (auto& candidate : context.nodes) {
    if (candidate.inst_name == inst_name
        && (candidate.kind == FastStaNodeKind::kBufferInput || candidate.kind == FastStaNodeKind::kBufferOutput)) {
      candidate.cell_master = target_master;
      if (candidate.kind == FastStaNodeKind::kBufferInput) {
        candidate.input_cap_pf = context.liberty_cell_by_master.at(target_master).input_cap_pf;
        candidate.max_slew_ns = context.liberty_cell_by_master.at(target_master).input_slew_limit_ns;
      }
    }
  }
  if (invalidate_context) {
    context.timing_valid = false;
    context.power_valid = false;
  }
  return input_node_id;
}

auto validateBufferMasterChange(const FastStaClockContext& context, const FastStaBufferMasterChange& change) -> bool
{
  if (change.node_id >= context.nodes.size()) {
    LOG_ERROR << "FastStaIncremental: buffer master change skipped because node id is invalid.";
    return false;
  }
  const auto& node = context.nodes.at(change.node_id);
  if (node.kind != FastStaNodeKind::kBufferInput && node.kind != FastStaNodeKind::kBufferOutput) {
    LOG_ERROR << "FastStaIncremental: node \"" << node.name << "\" is not a buffer node.";
    return false;
  }
  if (normalizeBufferInputNodeId(context, change.node_id) == kInvalidFastStaNodeId) {
    LOG_ERROR << "FastStaIncremental: buffer master change skipped because buffer input node is unavailable for \"" << node.name << "\".";
    return false;
  }
  if (change.cell_master.empty()) {
    LOG_ERROR << "FastStaIncremental: buffer master change skipped because target master is empty for \"" << node.name << "\".";
    return false;
  }
  return true;
}

}  // namespace

auto FastStaIncremental::changeBufferMaster(FastStaClockContext& context, FastStaNodeId node_id, std::string_view cell_master) -> bool
{
  return applyBufferMasterChange(context, node_id, cell_master, true) != kInvalidFastStaNodeId;
}

auto FastStaIncremental::changeBufferMasters(FastStaClockContext& context, const std::vector<FastStaBufferMasterChange>& changes) -> bool
{
  for (const auto& change : changes) {
    if (!validateBufferMasterChange(context, change)) {
      return false;
    }
  }
  for (const auto& change : changes) {
    if (applyBufferMasterChange(context, change.node_id, change.cell_master, false) == kInvalidFastStaNodeId) {
      context.timing_valid = false;
      context.power_valid = false;
      return false;
    }
  }
  context.timing_valid = false;
  context.power_valid = false;
  return true;
}

auto FastStaIncremental::changeBufferMasterIncremental(FastStaClockContext& context, FastStaNodeId node_id, std::string_view cell_master)
    -> std::optional<FastStaDirtyRegion>
{
  const auto input_node_id = applyBufferMasterChange(context, node_id, cell_master, false);
  if (input_node_id == kInvalidFastStaNodeId) {
    return std::nullopt;
  }
  return collectDirtyRegion(context, input_node_id);
}

}  // namespace icts
