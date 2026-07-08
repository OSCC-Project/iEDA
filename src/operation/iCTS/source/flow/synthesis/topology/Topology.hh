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
 * @file Topology.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief CTS topology formation entry for sink branches and source trunk.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Point.hh"
#include "design/Inst.hh"
#include "design/Net.hh"
#include "design/Pin.hh"
#include "module/topology/clustering/Clustering.hh"
#include "synthesis/htree/HTree.hh"

namespace icts {

class CharacterizationLibrary;
class Clock;
class Config;
class Design;
class DomainStatusTable;
class FastSTA;
class ClockLayout;
class SchemaWriter;
class Wrapper;
struct ClockDistributionContext;
struct SynthesisTraceSummary;

struct ClockTopologyInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
  Clock* clock = nullptr;
  std::size_t clock_index = 0U;
  ClockLayout* clock_layout = nullptr;
  SynthesisTraceSummary* summary = nullptr;
  DomainStatusTable* status_table = nullptr;
  CharacterizationLibrary* characterization_library = nullptr;
  std::size_t valid_sinks = 0U;
  const std::vector<ClockDistributionContext>* sink_domains = nullptr;
};

struct TopologyInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
  Net* root_net = nullptr;
  std::string object_name_prefix;
  CharacterizationLibrary* characterization_library = nullptr;
  std::vector<double> additional_characterization_lengths_um;
  double clock_period_ns = 0.0;
  std::string clock_period_source;
  HTree::LogContext log_context;
  bool htree_loads_are_local_buffers = false;
};

struct TopologyConfig
{
  std::optional<bool> enable_sink_clustering = std::nullopt;
};

class Topology
{
 public:
  struct ClusterBufferMeta
  {
    std::size_t cluster_index = 0;
    Point<int> location = Point<int>(0, 0);
    std::size_t sink_count = 0;
    Inst* inst = nullptr;
    Pin* input_pin = nullptr;
    Pin* output_pin = nullptr;
    Net* sink_net = nullptr;
  };

  struct ClusterLeafDistanceSummary
  {
    std::size_t count = 0;
    double min_distance_um = 0.0;
    double max_distance_um = 0.0;
    double mean_distance_um = 0.0;
    double median_distance_um = 0.0;
  };

  struct Output
  {
    HTree::Output htree_output;
    std::optional<ClusterOutput> cluster_output = std::nullopt;
    std::vector<ClusterBufferMeta> cluster_buffers;

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
    bool sink_clustering_enabled = false;
    std::optional<ClusterLeafDistanceSummary> cluster_leaf_distance_summary = std::nullopt;
    std::size_t selected_htree_level_count = 0;
    std::optional<unsigned> selected_htree_depth = std::nullopt;
    std::size_t htree_inserted_buffer_count = 0;
    std::size_t htree_inserted_net_count = 0;
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

  Topology() = delete;

  using Input = TopologyInput;
  using Config = TopologyConfig;

  static auto build(const Input& input, const Config& config) -> Build;
  static auto resetClockTopology(Clock& clock) -> void;
  static auto resetClockTopology(Design& design, Clock& clock) -> void;
  static auto formClock(const ClockTopologyInput& input) -> bool;
};

}  // namespace icts
