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
 * @file ClockLayout.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Readonly clock layout projection consumed by report and visualization steps.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "spatial/Point.hh"

namespace icts {

enum class LayoutNetRole
{
  kUnknown = 0,
  kClockSource = 1,
  kSourceToRoot = 2,
  kDownstream = 3,
  kSinkTree = 4
};

enum class LayoutInstRole
{
  kUnknown = 0,
  kLogicCell = 1,
  kClockSource = 2,
  kClockLoad = 3,
  kRootBuffer = 4,
  kHTreeBuffer = 5,
  kSourceRootBuffer = 6
};

enum class ClockLayoutPhase
{
  kUnknown,
  kReadData,
  kDownstreamHTree,
  kSourceToRootSegment,
  kSourceToRootHTree
};

enum class SinkDomainKind
{
  kUnknown,
  kHardMacro,
  kRegular,
  kSourceToRoot
};

enum class ClockLayoutMode
{
  kUnknown,
  kDesign,
  kFlyline
};

struct ClockLayoutSegment
{
  std::string clock_name;
  std::string net_name;
  Point<int> begin = Point<int>(-1, -1);
  Point<int> end = Point<int>(-1, -1);
  LayoutNetRole net_role = LayoutNetRole::kUnknown;
  SinkDomainKind sink_domain = SinkDomainKind::kUnknown;
  ClockLayoutPhase synthesis_phase = ClockLayoutPhase::kUnknown;
  std::size_t clock_index = 0U;
  int topology_depth = -1;
  int topology_level = -1;
  bool routed = true;
  bool degraded = false;
};

struct ClockLayoutNet
{
  std::string clock_name;
  std::string net_name;
  LayoutNetRole role = LayoutNetRole::kUnknown;
  SinkDomainKind sink_domain = SinkDomainKind::kUnknown;
  ClockLayoutPhase synthesis_phase = ClockLayoutPhase::kUnknown;
  std::size_t clock_index = 0U;
  int selected_depth = -1;
  int topology_level = -1;
  int topology_level_count = 0;
  std::vector<ClockLayoutSegment> routed_segments;
  std::vector<ClockLayoutSegment> flyline_segments;
};

struct ClockLayoutInst
{
  std::string clock_name;
  std::string inst_name;
  std::string cell_master;
  Point<int> origin = Point<int>(-1, -1);
  int32_t width_dbu = 0;
  int32_t height_dbu = 0;
  LayoutInstRole role = LayoutInstRole::kUnknown;
  SinkDomainKind sink_domain = SinkDomainKind::kUnknown;
  ClockLayoutPhase synthesis_phase = ClockLayoutPhase::kUnknown;
  std::size_t clock_index = 0U;
  int topology_depth = -1;
  int topology_level = -1;
};

struct ClockLayoutClock
{
  std::string clock_name;
  std::string clock_net_name;
  std::size_t clock_index = 0U;
  std::vector<ClockLayoutNet> nets;
  std::vector<ClockLayoutInst> insts;
};

class ClockLayout
{
 public:
  auto reset() -> void;
  auto set_design_dbu_per_um(int32_t dbu_per_um) -> void;
  auto get_design_dbu_per_um() const -> int32_t { return _design_dbu_per_um; }

  auto markSynthesisComplete(bool complete) -> void { _synthesis_complete = complete; }
  auto isSynthesisComplete() const -> bool { return _synthesis_complete; }
  auto markInstantiationDone(bool done) -> void { _instantiation_done = done; }
  auto isInstantiationDone() const -> bool { return _instantiation_done; }

  auto ensureClock(const std::string& clock_name, const std::string& clock_net_name, std::size_t clock_index) -> ClockLayoutClock&;
  auto findClock(std::size_t clock_index) -> ClockLayoutClock*;
  auto findClock(std::size_t clock_index) const -> const ClockLayoutClock*;
  auto findNet(std::size_t clock_index, const std::string& net_name) const -> const ClockLayoutNet*;
  auto findInst(std::size_t clock_index, const std::string& inst_name) const -> const ClockLayoutInst*;
  auto addNet(const ClockLayoutNet& layout_net) -> void;
  auto addInst(const ClockLayoutInst& layout_inst) -> void;

  auto get_clocks() const -> const std::vector<ClockLayoutClock>& { return _clocks; }
  auto get_clocks() -> std::vector<ClockLayoutClock>& { return _clocks; }

 private:
  std::vector<ClockLayoutClock> _clocks;
  int32_t _design_dbu_per_um = 1;
  bool _synthesis_complete = false;
  bool _instantiation_done = false;
};

auto ToString(LayoutNetRole role) -> const char*;
auto ToString(LayoutInstRole role) -> const char*;
auto ToString(ClockLayoutPhase stage) -> const char*;
auto ToString(SinkDomainKind domain) -> const char*;
auto ToString(ClockLayoutMode mode) -> const char*;

}  // namespace icts
