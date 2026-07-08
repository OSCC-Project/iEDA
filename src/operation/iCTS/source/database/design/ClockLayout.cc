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
 * @file ClockLayout.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Readonly clock layout projection store implementation.
 */

#include "design/ClockLayout.hh"

#include <algorithm>
#include <cstdint>
#include <string>

namespace icts {

auto ClockLayout::reset() -> void
{
  _clocks.clear();
  _design_dbu_per_um = 1;
  _synthesis_complete = false;
  _instantiation_done = false;
}

auto ClockLayout::set_design_dbu_per_um(int32_t dbu_per_um) -> void
{
  _design_dbu_per_um = std::max(dbu_per_um, int32_t{1});
}

auto ClockLayout::ensureClock(const std::string& clock_name, const std::string& clock_net_name, std::size_t clock_index)
    -> ClockLayoutClock&
{
  if (auto* layout_clock = findClock(clock_index); layout_clock != nullptr) {
    if (layout_clock->clock_name.empty()) {
      layout_clock->clock_name = clock_name;
    }
    if (layout_clock->clock_net_name.empty()) {
      layout_clock->clock_net_name = clock_net_name;
    }
    return *layout_clock;
  }

  _clocks.push_back(ClockLayoutClock{
      .clock_name = clock_name,
      .clock_net_name = clock_net_name,
      .clock_index = clock_index,
      .nets = {},
      .insts = {},
  });
  return _clocks.back();
}

auto ClockLayout::findClock(std::size_t clock_index) -> ClockLayoutClock*
{
  auto iter = std::ranges::find_if(
      _clocks, [clock_index](const ClockLayoutClock& layout_clock) -> bool { return layout_clock.clock_index == clock_index; });
  return iter == _clocks.end() ? nullptr : &(*iter);
}

auto ClockLayout::findClock(std::size_t clock_index) const -> const ClockLayoutClock*
{
  auto iter = std::ranges::find_if(
      _clocks, [clock_index](const ClockLayoutClock& layout_clock) -> bool { return layout_clock.clock_index == clock_index; });
  return iter == _clocks.end() ? nullptr : &(*iter);
}

auto ClockLayout::findNet(std::size_t clock_index, const std::string& net_name) const -> const ClockLayoutNet*
{
  auto clock_iter = std::ranges::find_if(
      _clocks, [clock_index](const ClockLayoutClock& layout_clock) -> bool { return layout_clock.clock_index == clock_index; });
  if (clock_iter == _clocks.end()) {
    return nullptr;
  }
  auto net_iter = std::ranges::find_if(clock_iter->nets,
                                       [&net_name](const ClockLayoutNet& layout_net) -> bool { return layout_net.net_name == net_name; });
  return net_iter == clock_iter->nets.end() ? nullptr : &(*net_iter);
}

auto ClockLayout::findInst(std::size_t clock_index, const std::string& inst_name) const -> const ClockLayoutInst*
{
  auto clock_iter = std::ranges::find_if(
      _clocks, [clock_index](const ClockLayoutClock& layout_clock) -> bool { return layout_clock.clock_index == clock_index; });
  if (clock_iter == _clocks.end()) {
    return nullptr;
  }
  auto inst_iter = std::ranges::find_if(
      clock_iter->insts, [&inst_name](const ClockLayoutInst& layout_inst) -> bool { return layout_inst.inst_name == inst_name; });
  return inst_iter == clock_iter->insts.end() ? nullptr : &(*inst_iter);
}

auto ClockLayout::addNet(const ClockLayoutNet& layout_net) -> void
{
  auto& layout_clock = ensureClock(layout_net.clock_name, "", layout_net.clock_index);
  layout_clock.nets.push_back(layout_net);
}

auto ClockLayout::addInst(const ClockLayoutInst& layout_inst) -> void
{
  auto& layout_clock = ensureClock(layout_inst.clock_name, "", layout_inst.clock_index);
  layout_clock.insts.push_back(layout_inst);
}

auto ToString(LayoutNetRole role) -> const char*
{
  switch (role) {
    case LayoutNetRole::kClockSource:
      return "clock_source";
    case LayoutNetRole::kDownstream:
      return "downstream";
    case LayoutNetRole::kSinkTree:
      return "sink_tree";
    case LayoutNetRole::kSourceToRoot:
      return "source_to_root";
    case LayoutNetRole::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto ToString(LayoutInstRole role) -> const char*
{
  switch (role) {
    case LayoutInstRole::kLogicCell:
      return "logic_cell";
    case LayoutInstRole::kClockSource:
      return "clock_source";
    case LayoutInstRole::kClockLoad:
      return "clock_load";
    case LayoutInstRole::kRootBuffer:
      return "root_buffer";
    case LayoutInstRole::kHTreeBuffer:
      return "htree_buffer";
    case LayoutInstRole::kSourceRootBuffer:
      return "source_root_buffer";
    case LayoutInstRole::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto ToString(ClockLayoutPhase stage) -> const char*
{
  switch (stage) {
    case ClockLayoutPhase::kReadData:
      return "read_data";
    case ClockLayoutPhase::kDownstreamHTree:
      return "downstream_htree";
    case ClockLayoutPhase::kSourceToRootSegment:
      return "source_to_root_segment";
    case ClockLayoutPhase::kSourceToRootHTree:
      return "source_to_root_htree";
    case ClockLayoutPhase::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto ToString(SinkDomainKind domain) -> const char*
{
  switch (domain) {
    case SinkDomainKind::kHardMacro:
      return "hard_macro";
    case SinkDomainKind::kRegular:
      return "regular";
    case SinkDomainKind::kSourceToRoot:
      return "source_to_root";
    case SinkDomainKind::kUnknown:
      return "unknown";
  }
  return "unknown";
}

auto ToString(ClockLayoutMode mode) -> const char*
{
  switch (mode) {
    case ClockLayoutMode::kDesign:
      return "design";
    case ClockLayoutMode::kFlyline:
      return "flyline";
    case ClockLayoutMode::kUnknown:
      return "unknown";
  }
  return "unknown";
}

}  // namespace icts
