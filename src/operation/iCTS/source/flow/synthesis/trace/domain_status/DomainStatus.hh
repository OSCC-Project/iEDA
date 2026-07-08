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
 * @file DomainStatusTable.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief Typed CTS clock-tree synthesis status row assembly.
 */

#pragma once

#include <cstddef>
#include <string>

#include "logger/Schema.hh"

namespace icts {

class Clock;
enum class SinkDomainKind;

enum class DomainStatus
{
  kSkipped,
  kFailed,
  kFinished
};

class DomainStatusTable
{
 public:
  explicit DomainStatusTable(TableRows& rows) : _rows(&rows) {}

  auto append(const Clock& clock, DomainStatus status, SinkDomainKind sink_domain, std::size_t valid_sinks, std::size_t sink_domain_sinks,
              const std::string& detail) -> void;
  auto appendNoDomain(const Clock& clock, DomainStatus status, std::size_t valid_sinks, std::size_t sink_domain_sinks,
                      const std::string& detail) -> void;
  auto appendNullClock(DomainStatus status, const std::string& detail) -> void;

 private:
  TableRows* _rows = nullptr;
};

auto ToString(DomainStatus status) -> const char*;

}  // namespace icts
