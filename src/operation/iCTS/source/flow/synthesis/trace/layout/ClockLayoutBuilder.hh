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
 * @file ClockLayoutBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief CTS clock layout projection construction helper.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "design/ClockLayout.hh"

namespace icts {

class Clock;
class Inst;
class Net;
class Pin;
struct SinkDomainSynthesisTopology;
struct SourceToRootSynthesisTopology;

struct SinkDomainLayoutAnchor
{
  SinkDomainKind sink_domain = SinkDomainKind::kUnknown;
  const Inst* root_buffer = nullptr;
  const Net* downstream_net = nullptr;
};

class ClockLayoutBuilder
{
 public:
  ClockLayoutBuilder() = delete;

  static auto appendSinkInsts(ClockLayout& clock_layout, const Clock& clock, std::size_t clock_index, const std::vector<Pin*>& sinks,
                              SinkDomainKind sink_domain) -> void;
  static auto appendDirectSinkDomain(ClockLayout& clock_layout, const Clock& clock, std::size_t clock_index,
                                     const SinkDomainLayoutAnchor& sink_domain_anchor) -> void;
  static auto makeSinkDomainLayout(const Clock& clock, std::size_t clock_index, const SinkDomainLayoutAnchor& sink_domain_anchor,
                                   const SinkDomainSynthesisTopology& layout_topology) -> ClockLayout;
  static auto makeSourceToRootLayout(const Clock& clock, std::size_t clock_index, const Net& source_net,
                                     const SourceToRootSynthesisTopology& layout_topology, ClockLayoutPhase synthesis_phase) -> ClockLayout;
  static auto merge(ClockLayout& target, const ClockLayout& source) -> void;
};

}  // namespace icts
