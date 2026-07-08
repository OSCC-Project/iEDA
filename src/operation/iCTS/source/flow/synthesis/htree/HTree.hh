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
 * @file HTree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-14
 * @brief H-tree topology-family synthesis entry.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "characterization/HTreeTopologyChar.hh"
#include "characterization/HTreeTopologyPattern.hh"
#include "characterization/PatternId.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "module/characterization/Characterization.hh"
#include "spatial/Point.hh"
#include "spatial/Tree.hh"

namespace icts {

class CharacterizationLibrary;
class Design;
class SchemaWriter;
class Wrapper;

class HTree
{
 public:
  enum class LoadRole
  {
    kSink,
    kLocalBuffer
  };

  struct LogContext
  {
    std::string clock_name;
    std::string clock_net_name;
    std::string sink_domain;
    std::string stage;
    std::string object_name_prefix;
  };

  struct Input
  {
    Net* root_net = nullptr;
    Design* design = nullptr;
    Wrapper* wrapper = nullptr;
    SchemaWriter* reporter = nullptr;
    CharacterizationLibrary* characterization_library = nullptr;
    CharBuilder::Input characterization_input;
    CharBuilder::Config characterization_config;
    std::vector<double> additional_characterization_lengths_um;
    std::optional<Point<int>> fixed_topology_root_location = std::nullopt;
    double clock_period_ns = 0.0;
    std::string clock_period_source;
    LogContext log_context;
    std::string object_name_prefix;
    LoadRole load_role = LoadRole::kSink;
  };

  struct Config
  {
    bool force_branch_buffer = false;
    std::optional<double> min_top_input_slew_ns = std::nullopt;
    std::optional<unsigned> target_depth = std::nullopt;
    unsigned depth_explore_window = 1U;
    double topology_tolerance = 0.0;
    std::size_t max_fanout = 0U;
    bool has_max_cap = false;
    double max_cap_pf = 0.0;
    bool enable_root_driver_sizing = true;
    bool allow_boundary_relaxation = false;
    bool enable_analytical_solver = false;
    int routing_layer = 0;
    std::optional<double> wire_width_um = std::nullopt;
  };

  struct LevelPlan
  {
    int requested_length_dbu = 0;
    double requested_length_um = 0.0;
    unsigned aligned_length_idx = 0;
    double aligned_length_um = 0.0;
    bool is_leaf_level = false;
    bool selected_has_any_buffer = false;
    bool selected_has_terminal_branch_buffer = false;
    std::string selected_leaf_buffer_cell_master;
    std::string selected_terminal_cell_master;
    std::size_t selected_buffer_count = 0U;
    double selected_buffer_area_um2 = 0.0;
    std::size_t selected_weighted_buffer_count = 0U;
    double selected_weighted_buffer_area_um2 = 0.0;
    PatternId segment_pattern_id = PatternId::segment(0);
  };

  struct InsertedInstLevel
  {
    Inst* inst = nullptr;
    int topology_level = -1;
    std::size_t index_in_level = 0U;
  };

  struct InsertedNetLevel
  {
    Net* net = nullptr;
    int topology_level = -1;
    std::size_t index_in_level = 0U;
  };

  struct Output
  {
    Tree topology;
    std::vector<LevelPlan> levels;
    std::optional<HTreeTopologyChar> best_char = std::nullopt;
    std::optional<HTreeTopologyPattern> best_pattern = std::nullopt;

    std::vector<std::unique_ptr<Inst>> inserted_insts;
    std::vector<std::unique_ptr<Pin>> inserted_pins;
    std::vector<std::unique_ptr<Net>> inserted_nets;
    std::vector<InsertedInstLevel> inserted_inst_levels;
    std::vector<InsertedNetLevel> inserted_net_levels;

    Inst* root_inst = nullptr;
    Pin* root_input_pin = nullptr;
    Pin* root_output_pin = nullptr;
    Net* root_net = nullptr;
  };

  struct Summary
  {
    bool success = false;
    std::string failure_reason;
    std::optional<unsigned> selected_depth = std::nullopt;
    bool used_boundary_relaxation = false;
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

  HTree() = delete;

  static auto build(const Input& input, const Config& config) -> Build;
};

}  // namespace icts
