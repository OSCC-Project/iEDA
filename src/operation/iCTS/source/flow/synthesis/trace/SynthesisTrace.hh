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
 * @file SynthesisTrace.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS synthesis execution trace records.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace icts {

enum class SynthesisOutcome
{
  kFinished,
  kFailed,
  kNoOp
};

struct SynthesisTraceStatusRecord
{
  std::string clock_name;
  std::string clock_net_name;
  std::string status;
  std::string sink_domain;
  std::size_t valid_sink_count = 0U;
  std::size_t sink_domain_sink_count = 0U;
  std::string detail;
};

struct SynthesisTraceSummary
{
  bool success = true;
  SynthesisOutcome outcome = SynthesisOutcome::kFinished;
  std::string no_op_reason;
  std::size_t total_clocks = 0U;
  std::size_t successful_clocks = 0U;
  std::size_t skipped_clocks = 0U;
  std::size_t failed_clocks = 0U;
  std::size_t total_sink_domains = 0U;
  std::size_t hard_macro_sinks = 0U;
  std::size_t regular_sinks = 0U;
  std::size_t selected_htree_level_count = 0U;
  unsigned selected_htree_depth = 0U;
  std::size_t htree_inserted_buffer_count = 0U;
  std::size_t htree_inserted_net_count = 0U;
  std::vector<SynthesisTraceStatusRecord> domain_status;
};

class SynthesisTrace
{
 public:
  SynthesisTrace() = delete;

  static auto reset(SynthesisTraceSummary& trace) -> void;
};

}  // namespace icts
