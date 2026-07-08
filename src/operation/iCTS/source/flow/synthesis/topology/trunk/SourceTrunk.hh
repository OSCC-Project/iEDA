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
 * @file SourceTrunk.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Dispatches Topology source-to-root segment or HTree construction.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/topology/SourceTrunkStage.hh"

namespace icts {

class CharacterizationLibrary;
class Config;
class Design;
class FastSTA;
class SchemaWriter;
class Wrapper;

namespace topology {

struct SourceTrunkInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
  Net* source_net = nullptr;
  Pin* clock_source = nullptr;
  std::vector<Pin*> root_inputs;
  std::string object_name_prefix;
  CharacterizationLibrary* characterization_library = nullptr;
  double clock_period_ns = 0.0;
  std::string clock_period_source;
  HTree::LogContext log_context;
};

struct SourceTrunkOutput
{
  HTree::Output htree_output;

  std::vector<std::unique_ptr<Inst>> inserted_insts;
  std::vector<std::unique_ptr<Pin>> inserted_pins;
  std::vector<std::unique_ptr<Net>> inserted_nets;
  std::vector<HTree::InsertedInstLevel> inserted_inst_levels;
  std::vector<HTree::InsertedNetLevel> inserted_net_levels;
};

struct SourceTrunkSummary
{
  bool success = false;
  std::string failure_reason;
  SourceTrunkStage stage = SourceTrunkStage::kUnknown;
  std::optional<unsigned> selected_depth = std::nullopt;
  std::size_t selected_level_count = 0U;
  std::size_t inserted_buffer_count = 0U;
  std::size_t inserted_net_count = 0U;
  bool used_boundary_relaxation = false;
};

struct SourceTrunkBuild
{
  SourceTrunkBuild() = default;

  SourceTrunkBuild(const SourceTrunkBuild&) = delete;
  auto operator=(const SourceTrunkBuild&) -> SourceTrunkBuild& = delete;

  SourceTrunkBuild(SourceTrunkBuild&& rhs) noexcept = default;
  auto operator=(SourceTrunkBuild&& rhs) noexcept -> SourceTrunkBuild& = default;

  SourceTrunkOutput output;
  SourceTrunkSummary summary;
};

auto BuildSourceTrunkTree(const SourceTrunkInput& input) -> SourceTrunkBuild;

}  // namespace topology
}  // namespace icts
