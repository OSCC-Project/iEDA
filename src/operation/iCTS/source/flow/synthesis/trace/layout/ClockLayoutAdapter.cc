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
 * @file ClockLayoutAdapter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief Clock-tree synthesis-to-layout adapter implementation.
 */

#include "synthesis/trace/layout/ClockLayoutAdapter.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "ClockLayout.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/topology/trunk/SourceTrunk.hh"

namespace icts {

class Inst;
class Net;

namespace {

auto selectedDepthToInt(const std::optional<unsigned>& selected_depth) -> int
{
  return selected_depth.has_value() ? static_cast<int>(*selected_depth) : -1;
}

auto findInsertedInstLevel(const std::vector<HTree::InsertedInstLevel>& levels, const Inst* inst) -> int
{
  if (inst == nullptr) {
    return -1;
  }
  const auto iter = std::ranges::find_if(levels, [inst](const HTree::InsertedInstLevel& level) -> bool { return level.inst == inst; });
  return iter == levels.end() ? -1 : iter->topology_level;
}

auto findInsertedNetLevel(const std::vector<HTree::InsertedNetLevel>& levels, const Net* net) -> std::optional<int>
{
  if (net == nullptr) {
    return std::nullopt;
  }
  const auto iter = std::ranges::find_if(levels, [net](const HTree::InsertedNetLevel& level) -> bool { return level.net == net; });
  if (iter == levels.end()) {
    return std::nullopt;
  }
  return iter->topology_level;
}

auto inferNetTopologyLevel(LayoutNetRole role, ClockLayoutPhase synthesis_phase, int selected_depth) -> int
{
  if (role == LayoutNetRole::kSinkTree) {
    return selected_depth >= 0 ? selected_depth : 0;
  }
  if (role == LayoutNetRole::kSourceToRoot && synthesis_phase == ClockLayoutPhase::kSourceToRootHTree) {
    return selected_depth >= 0 ? selected_depth : 0;
  }
  return 0;
}

auto resolveNetTopologyLevel(const std::vector<HTree::InsertedNetLevel>& levels, const Net* net, LayoutNetRole role,
                             ClockLayoutPhase synthesis_phase, int selected_depth) -> int
{
  const auto inserted_level = findInsertedNetLevel(levels, net);
  if (inserted_level.has_value()) {
    return *inserted_level;
  }
  return inferNetTopologyLevel(role, synthesis_phase, selected_depth);
}

}  // namespace

auto ClockLayoutAdapter::makeSinkDomainLayoutTopology(const Topology::Build& result) -> SinkDomainSynthesisTopology
{
  SinkDomainSynthesisTopology layout_topology{
      .selected_depth = selectedDepthToInt(result.summary.selected_htree_depth),
      .topology_level_count = static_cast<int>(result.summary.selected_htree_level_count),
      .inserted_insts = {},
      .inserted_nets = {},
  };
  layout_topology.inserted_insts.reserve(result.output.inserted_insts.size());
  for (const auto& inst : result.output.inserted_insts) {
    if (inst == nullptr) {
      continue;
    }
    layout_topology.inserted_insts.push_back(ClockLayoutInstTopology{
        .inst = inst.get(),
        .topology_level = findInsertedInstLevel(result.output.inserted_inst_levels, inst.get()),
    });
  }
  layout_topology.inserted_nets.reserve(result.output.inserted_nets.size());
  for (const auto& net : result.output.inserted_nets) {
    if (net == nullptr) {
      continue;
    }
    layout_topology.inserted_nets.push_back(ClockLayoutNetTopology{
        .net = net.get(),
        .topology_level = resolveNetTopologyLevel(result.output.inserted_net_levels, net.get(), LayoutNetRole::kSinkTree,
                                                  ClockLayoutPhase::kDownstreamHTree, layout_topology.selected_depth),
    });
  }
  return layout_topology;
}

auto ClockLayoutAdapter::makeSourceTrunkLayoutTopology(const topology::SourceTrunkBuild& result, ClockLayoutPhase synthesis_phase)
    -> SourceToRootSynthesisTopology
{
  SourceToRootSynthesisTopology layout_topology{
      .selected_depth = selectedDepthToInt(result.summary.selected_depth),
      .topology_level_count = static_cast<int>(result.summary.selected_level_count),
      .inserted_insts = {},
      .inserted_nets = {},
  };
  layout_topology.inserted_insts.reserve(result.output.inserted_insts.size());
  for (const auto& inst : result.output.inserted_insts) {
    if (inst == nullptr) {
      continue;
    }
    layout_topology.inserted_insts.push_back(ClockLayoutInstTopology{
        .inst = inst.get(),
        .topology_level = findInsertedInstLevel(result.output.inserted_inst_levels, inst.get()),
    });
  }
  layout_topology.inserted_nets.reserve(result.output.inserted_nets.size());
  for (const auto& net : result.output.inserted_nets) {
    if (net == nullptr) {
      continue;
    }
    layout_topology.inserted_nets.push_back(ClockLayoutNetTopology{
        .net = net.get(),
        .topology_level = resolveNetTopologyLevel(result.output.inserted_net_levels, net.get(), LayoutNetRole::kSourceToRoot,
                                                  synthesis_phase, layout_topology.selected_depth),
    });
  }
  return layout_topology;
}

}  // namespace icts
