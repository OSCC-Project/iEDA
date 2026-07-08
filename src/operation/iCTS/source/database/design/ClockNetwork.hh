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
 * @file ClockNetwork.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief Stable CTS clock-tree semantic model over design-owned objects.
 */

#pragma once

#include <string>
#include <vector>

namespace icts {

class Clock;
class Inst;
class Net;
class Pin;

class ClockNetwork
{
 public:
  enum class DomainKind
  {
    kUnknown,
    kHardMacro,
    kRegular,
    kSourceToRoot
  };

  enum class InstRole
  {
    kUnknown,
    kClockSource,
    kClockLoad,
    kRootBuffer,
    kTreeBuffer,
    kSourceRootBuffer
  };

  enum class NetRole
  {
    kUnknown,
    kClockSource,
    kSourceToRoot,
    kDownstream,
    kSinkTree
  };

  enum class TopologyKind
  {
    kUnknown,
    kDirect,
    kHTree,
    kSourceToRootSegment,
    kSourceToRootHTree
  };

  enum class LifecycleState
  {
    kEmpty,
    kPlanned,
    kInstantiated,
    kProjectedToExternalDb
  };

  struct TopologyRecord
  {
    TopologyKind kind = TopologyKind::kUnknown;
    unsigned selected_depth = 0U;
    unsigned level_count = 0U;
  };

  struct DomainRecord
  {
    DomainKind kind = DomainKind::kUnknown;
    std::vector<Pin*> sinks;
    Inst* root_buffer = nullptr;
    Pin* root_input = nullptr;
    Pin* root_output = nullptr;
    Net* downstream_net = nullptr;
    TopologyRecord topology;
  };

  struct InstRecord
  {
    Inst* inst = nullptr;
    InstRole role = InstRole::kUnknown;
    DomainKind domain = DomainKind::kUnknown;
    int topology_level = -1;
  };

  struct NetRecord
  {
    Net* net = nullptr;
    NetRole role = NetRole::kUnknown;
    DomainKind domain = DomainKind::kUnknown;
    int topology_level = -1;
  };

  auto reset() -> void;
  auto bindClock(Clock* clock, std::string clock_name, std::string clock_net_name) -> void;
  auto set_state(LifecycleState state) -> void { _state = state; }

  auto ensureDomain(DomainKind kind) -> DomainRecord&;
  auto recordDomainRoot(DomainKind kind, Inst* root_buffer, Pin* root_input, Pin* root_output, Net* downstream_net) -> void;
  auto recordDomainSinks(DomainKind kind, const std::vector<Pin*>& sinks) -> void;
  auto recordDomainTopology(DomainKind kind, const TopologyRecord& topology) -> void;
  auto recordInstRole(Inst* inst, InstRole role, DomainKind domain, int topology_level = -1) -> void;
  auto recordNetRole(Net* net, NetRole role, DomainKind domain, int topology_level = -1) -> void;

  auto findDomain(DomainKind kind) -> DomainRecord*;
  auto findDomain(DomainKind kind) const -> const DomainRecord*;
  auto findInstRecord(const Inst* inst) const -> const InstRecord*;
  auto findNetRecord(const Net* net) const -> const NetRecord*;

  auto get_clock() const -> Clock* { return _clock; }
  auto get_clock_name() const -> const std::string& { return _clock_name; }
  auto get_clock_net_name() const -> const std::string& { return _clock_net_name; }
  auto get_state() const -> LifecycleState { return _state; }
  auto get_domains() const -> const std::vector<DomainRecord>& { return _domains; }
  auto get_insts() const -> const std::vector<InstRecord>& { return _insts; }
  auto get_nets() const -> const std::vector<NetRecord>& { return _nets; }

 private:
  Clock* _clock = nullptr;
  std::string _clock_name;
  std::string _clock_net_name;
  LifecycleState _state = LifecycleState::kEmpty;
  std::vector<DomainRecord> _domains;
  std::vector<InstRecord> _insts;
  std::vector<NetRecord> _nets;
};

}  // namespace icts
