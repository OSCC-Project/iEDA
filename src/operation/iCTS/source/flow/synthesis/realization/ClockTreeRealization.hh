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
 * @file ClockTreeRealization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-26
 * @brief Clock-tree realization helper for committing synthesized CTS objects.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace icts {

class Clock;
class Design;
class Inst;
class Net;
class Pin;
class Wrapper;
enum class SinkDomainKind;

struct ClockSinkPartitionOutput
{
  std::vector<Pin*> macro_sinks;
  std::vector<Pin*> regular_sinks;
};

struct NetConnectionInput
{
  Net* net = nullptr;
  Pin* driver = nullptr;
  std::vector<Pin*> loads;
};

struct SourceToRootNetReuseInput
{
  Clock* clock = nullptr;
  Pin* clock_source = nullptr;
  std::vector<Pin*> root_buffer_inputs;
};

struct SinkDomainRootBufferInput
{
  Design* design = nullptr;
  Clock* clock = nullptr;
  std::string domain_prefix;
  std::vector<Pin*> sinks;
  std::string cell_master;
  std::string input_pin_name;
  std::string output_pin_name;
};

struct SinkDomainRootBufferOutput
{
  Inst* root_buffer = nullptr;
  Pin* root_input = nullptr;
  Pin* root_output = nullptr;
};

struct SinkDomainRootBufferSelectionInput
{
  Design* design = nullptr;
  Clock* clock = nullptr;
  Wrapper* wrapper = nullptr;
  std::string domain_prefix;
  std::vector<std::string> buffer_types;
  std::vector<Pin*> sinks;
};

struct SinkDomainDownstreamNetInput
{
  Design* design = nullptr;
  Clock* clock = nullptr;
  std::string domain_prefix;
  Pin* root_output = nullptr;
  std::vector<Pin*> sinks;
};

struct InsertedObjectCommitInput
{
  Design* design = nullptr;
  Clock* clock = nullptr;
  std::vector<std::unique_ptr<Inst>>* inserted_insts = nullptr;
  std::vector<std::unique_ptr<Pin>>* inserted_pins = nullptr;
  std::vector<std::unique_ptr<Net>>* inserted_nets = nullptr;
};

class ClockTreeRealization
{
 public:
  ClockTreeRealization() = delete;

  static auto partitionClockSinks(const std::vector<Pin*>& sinks) -> ClockSinkPartitionOutput;
  static auto makeSinkDomainPrefix(const Clock& clock, std::size_t clock_index, SinkDomainKind sink_domain) -> std::string;
  static auto addRootBufferForSinkDomain(const SinkDomainRootBufferSelectionInput& input) -> SinkDomainRootBufferOutput;
  static auto addRootBufferForSinkDomain(const SinkDomainRootBufferInput& input) -> SinkDomainRootBufferOutput;
  static auto reconnectNet(const NetConnectionInput& input) -> void;
  static auto connectSinkDomainDownstreamNet(const SinkDomainDownstreamNetInput& input) -> Net*;
  static auto restoreClockSourceNetToClockLoads(Clock& clock) -> void;
  static auto reuseClockSourceNetAsSourceToRootBuffers(const SourceToRootNetReuseInput& input) -> Net*;
  static auto commitInsertedObjects(const InsertedObjectCommitInput& input) -> bool;
};

}  // namespace icts
