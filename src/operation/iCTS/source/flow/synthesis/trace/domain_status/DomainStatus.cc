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
 * @file DomainStatusTable.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief Typed CTS clock-tree synthesis status row assembly implementation.
 */

#include "synthesis/trace/domain_status/DomainStatus.hh"

#include <string>
#include <vector>

#include "ClockLayout.hh"
#include "design/Clock.hh"

namespace icts {
namespace {

auto sinkDomainLabel(SinkDomainKind sink_domain) -> const char*
{
  if (sink_domain == SinkDomainKind::kUnknown) {
    return "none";
  }
  return ToString(sink_domain);
}

}  // namespace

auto ToString(DomainStatus status) -> const char*
{
  switch (status) {
    case DomainStatus::kSkipped:
      return "skipped";
    case DomainStatus::kFailed:
      return "failed";
    case DomainStatus::kFinished:
      return "finished";
  }
  return "failed";
}

auto DomainStatusTable::append(const Clock& clock, DomainStatus status, SinkDomainKind sink_domain, std::size_t valid_sinks,
                               std::size_t sink_domain_sinks, const std::string& detail) -> void
{
  _rows->push_back({
      clock.get_clock_name(),
      clock.get_clock_net_name(),
      ToString(status),
      sinkDomainLabel(sink_domain),
      std::to_string(valid_sinks),
      std::to_string(sink_domain_sinks),
      detail,
  });
}

auto DomainStatusTable::appendNoDomain(const Clock& clock, DomainStatus status, std::size_t valid_sinks, std::size_t sink_domain_sinks,
                                       const std::string& detail) -> void
{
  append(clock, status, SinkDomainKind::kUnknown, valid_sinks, sink_domain_sinks, detail);
}

auto DomainStatusTable::appendNullClock(DomainStatus status, const std::string& detail) -> void
{
  _rows->push_back({"", "", ToString(status), "none", "0", "0", detail});
}

}  // namespace icts
