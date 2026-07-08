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
 * @file ClockDistribution.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS clock source, root, and sink-domain distribution preparation implementation.
 */

#include "synthesis/distribution/ClockDistribution.hh"

#include <glog/logging.h>

#include <ostream>
#include <utility>

#include "ClockLayout.hh"
#include "Log.hh"
#include "design/Clock.hh"
#include "synthesis/realization/ClockTreeRealization.hh"
#include "synthesis/trace/domain_status/DomainStatus.hh"

namespace icts {
namespace {

auto addRootBuffer(const ClockDistributionInput& input, const std::string& domain_prefix) -> SinkDomainRootBufferOutput
{
  if (input.root_buffer_spec == nullptr) {
    return ClockTreeRealization::addRootBufferForSinkDomain(SinkDomainRootBufferSelectionInput{
        .design = input.design,
        .clock = input.clock,
        .wrapper = input.wrapper,
        .domain_prefix = domain_prefix,
        .buffer_types = input.root_buffer_types,
        .sinks = input.sinks,
    });
  }
  return ClockTreeRealization::addRootBufferForSinkDomain(SinkDomainRootBufferInput{
      .design = input.design,
      .clock = input.clock,
      .domain_prefix = domain_prefix,
      .sinks = input.sinks,
      .cell_master = input.root_buffer_spec->cell_master,
      .input_pin_name = input.root_buffer_spec->input_pin_name,
      .output_pin_name = input.root_buffer_spec->output_pin_name,
  });
}

}  // namespace

auto ClockDistribution::partitionSinkDomains(const Clock& clock) -> ClockDistributionPartition
{
  ClockDistributionPartition partition;
  auto sink_partition = ClockTreeRealization::partitionClockSinks(clock.get_loads());
  partition.macro_sinks = std::move(sink_partition.macro_sinks);
  partition.regular_sinks = std::move(sink_partition.regular_sinks);
  partition.valid_sink_count = partition.macro_sinks.size() + partition.regular_sinks.size();
  return partition;
}

auto ClockDistribution::prepare(const ClockDistributionInput& input) -> std::optional<ClockDistributionContext>
{
  LOG_FATAL_IF(input.design == nullptr) << "ClockDistribution: design is null.";
  LOG_FATAL_IF(input.clock == nullptr) << "ClockDistribution: clock is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "ClockDistribution: Wrapper is null.";
  LOG_FATAL_IF(input.status_table == nullptr) << "ClockDistribution: status table is null.";

  if (input.sinks.empty()) {
    return ClockDistributionContext{};
  }

  auto& design = *input.design;
  auto& clock = *input.clock;
  auto& status_table = *input.status_table;
  const auto* const sink_domain_label = ToString(input.sink_domain);
  ClockDistributionContext context;
  context.sink_domain = input.sink_domain;
  context.domain_prefix = ClockTreeRealization::makeSinkDomainPrefix(clock, input.clock_index, input.sink_domain);
  context.sinks = input.sinks;

  const auto root_buffer_output = addRootBuffer(input, context.domain_prefix);
  context.root_buffer = root_buffer_output.root_buffer;
  context.root_input = root_buffer_output.root_input;
  context.root_output = root_buffer_output.root_output;
  if (context.root_buffer == nullptr || context.root_input == nullptr || context.root_output == nullptr) {
    status_table.append(clock, DomainStatus::kFailed, input.sink_domain, input.valid_sinks, input.sinks.size(),
                        "failed to insert root buffer");
    LOG_ERROR << "ClockDistribution: clock \"" << clock.get_clock_name() << "\" sink domain " << sink_domain_label
              << " failed because root-buffer insertion failed.";
    return std::nullopt;
  }

  context.downstream_net = ClockTreeRealization::connectSinkDomainDownstreamNet(SinkDomainDownstreamNetInput{
      .design = &design,
      .clock = &clock,
      .domain_prefix = context.domain_prefix,
      .root_output = context.root_output,
      .sinks = input.sinks,
  });
  if (context.downstream_net == nullptr) {
    status_table.append(clock, DomainStatus::kFailed, input.sink_domain, input.valid_sinks, input.sinks.size(),
                        "failed to create downstream net");
    LOG_ERROR << "ClockDistribution: clock \"" << clock.get_clock_name() << "\" sink domain " << sink_domain_label
              << " failed because downstream net creation failed.";
    return std::nullopt;
  }
  return context;
}

}  // namespace icts
