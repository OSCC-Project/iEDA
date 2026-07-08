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
 * @file ClockNetwork.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief Stable CTS clock-tree semantic model over design-owned objects implementation.
 */

#include "ClockNetwork.hh"

#include <algorithm>
#include <utility>

namespace icts {

auto ClockNetwork::reset() -> void
{
  _clock = nullptr;
  _clock_name.clear();
  _clock_net_name.clear();
  _state = LifecycleState::kEmpty;
  _domains.clear();
  _insts.clear();
  _nets.clear();
}

auto ClockNetwork::bindClock(Clock* clock, std::string clock_name, std::string clock_net_name) -> void
{
  _clock = clock;
  _clock_name = std::move(clock_name);
  _clock_net_name = std::move(clock_net_name);
}

auto ClockNetwork::ensureDomain(DomainKind kind) -> DomainRecord&
{
  auto* domain = findDomain(kind);
  if (domain != nullptr) {
    return *domain;
  }
  _domains.push_back(DomainRecord{
      .kind = kind,
      .sinks = {},
      .root_buffer = nullptr,
      .root_input = nullptr,
      .root_output = nullptr,
      .downstream_net = nullptr,
      .topology = {},
  });
  return _domains.back();
}

auto ClockNetwork::recordDomainRoot(DomainKind kind, Inst* root_buffer, Pin* root_input, Pin* root_output, Net* downstream_net) -> void
{
  auto& domain = ensureDomain(kind);
  domain.root_buffer = root_buffer;
  domain.root_input = root_input;
  domain.root_output = root_output;
  domain.downstream_net = downstream_net;
}

auto ClockNetwork::recordDomainSinks(DomainKind kind, const std::vector<Pin*>& sinks) -> void
{
  ensureDomain(kind).sinks = sinks;
}

auto ClockNetwork::recordDomainTopology(DomainKind kind, const TopologyRecord& topology) -> void
{
  ensureDomain(kind).topology = topology;
}

auto ClockNetwork::recordInstRole(Inst* inst, InstRole role, DomainKind domain, int topology_level) -> void
{
  if (inst == nullptr) {
    return;
  }
  auto iter = std::ranges::find_if(_insts, [inst](const InstRecord& record) -> bool { return record.inst == inst; });
  if (iter != _insts.end()) {
    *iter = InstRecord{.inst = inst, .role = role, .domain = domain, .topology_level = topology_level};
    return;
  }
  _insts.push_back(InstRecord{.inst = inst, .role = role, .domain = domain, .topology_level = topology_level});
}

auto ClockNetwork::recordNetRole(Net* net, NetRole role, DomainKind domain, int topology_level) -> void
{
  if (net == nullptr) {
    return;
  }
  auto iter = std::ranges::find_if(_nets, [net](const NetRecord& record) -> bool { return record.net == net; });
  if (iter != _nets.end()) {
    *iter = NetRecord{.net = net, .role = role, .domain = domain, .topology_level = topology_level};
    return;
  }
  _nets.push_back(NetRecord{.net = net, .role = role, .domain = domain, .topology_level = topology_level});
}

auto ClockNetwork::findDomain(DomainKind kind) -> DomainRecord*
{
  auto iter = std::ranges::find_if(_domains, [kind](const DomainRecord& domain) -> bool { return domain.kind == kind; });
  return iter == _domains.end() ? nullptr : &(*iter);
}

auto ClockNetwork::findDomain(DomainKind kind) const -> const DomainRecord*
{
  auto iter = std::ranges::find_if(_domains, [kind](const DomainRecord& domain) -> bool { return domain.kind == kind; });
  return iter == _domains.end() ? nullptr : &(*iter);
}

auto ClockNetwork::findInstRecord(const Inst* inst) const -> const InstRecord*
{
  auto iter = std::ranges::find_if(_insts, [inst](const InstRecord& record) -> bool { return record.inst == inst; });
  return iter == _insts.end() ? nullptr : &(*iter);
}

auto ClockNetwork::findNetRecord(const Net* net) const -> const NetRecord*
{
  auto iter = std::ranges::find_if(_nets, [net](const NetRecord& record) -> bool { return record.net == net; });
  return iter == _nets.end() ? nullptr : &(*iter);
}

}  // namespace icts
