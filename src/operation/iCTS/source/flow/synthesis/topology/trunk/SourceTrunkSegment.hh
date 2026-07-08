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
 * @file SourceTrunkSegment.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Source-to-sink segment synthesis using reusable characterization frontiers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "SegmentChar.hh"
#include "characterization/Characterization.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "synthesis/htree/HTree.hh"

namespace icts {

class CharacterizationLibrary;
class SchemaWriter;
class Wrapper;
class SourceTrunkSegment
{
 public:
  struct Input
  {
    Net* source_net = nullptr;
    Pin* source = nullptr;
    Pin* sink = nullptr;
    CharacterizationLibrary* characterization_library = nullptr;
    Wrapper* wrapper = nullptr;
    CharBuilder::Input characterization_input;
    CharBuilder::Config characterization_config;
    SchemaWriter* reporter = nullptr;
    std::int32_t dbu_per_um = 0;
    double required_load_cap_pf = 0.0;
    double source_drive_cap_pf = 0.0;
    std::string object_name_prefix;
    HTree::LogContext log_context;
  };

  struct Config
  {
    std::optional<double> min_input_slew_ns = std::nullopt;
  };

  struct Output
  {
    std::optional<SegmentChar> best_char = std::nullopt;

    std::vector<std::unique_ptr<Inst>> inserted_insts;
    std::vector<std::unique_ptr<Pin>> inserted_pins;
    std::vector<std::unique_ptr<Net>> inserted_nets;
    std::vector<HTree::InsertedInstLevel> inserted_inst_levels;
    std::vector<HTree::InsertedNetLevel> inserted_net_levels;
  };

  struct Summary
  {
    bool success = false;
    std::string failure_reason;
    bool used_boundary_relaxation = false;
    std::string boundary_relaxation_reason;
    std::size_t strict_candidate_count = 0U;
    std::size_t relaxed_candidate_count = 0U;
    double length_um = 0.0;
    unsigned length_idx = 0U;
    unsigned required_load_cap_idx = 0U;
    unsigned source_drive_cap_idx = 0U;
    std::optional<unsigned> min_input_slew_idx = std::nullopt;
  };

  struct Build
  {
    Build() = default;

    Build(const Build&) = delete;
    auto operator=(const Build&) -> Build& = delete;

    Build(Build&& rhs) noexcept = default;
    auto operator=(Build&& rhs) noexcept -> Build& = default;

    Output output;
    Summary summary;
  };

  SourceTrunkSegment() = delete;

  static auto build(const Input& input, const Config& config) -> Build;
};

}  // namespace icts
