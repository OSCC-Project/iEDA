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
 * @file TopologyGen.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief Topology generator for CTS.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Point.hh"
#include "Tree.hh"
#include "config/TopologyConfig.hh"

namespace icts {

class Pin;
class SchemaWriter;
enum class TopologyGenLoadCountKind
{
  kSink,
  kLocalBuffer
};

struct TopologyGenInput
{
  std::optional<Point<int>> fixed_root_location = std::nullopt;
  int32_t dbu_per_um = 1;
  TopologyGenLoadCountKind load_count_kind = TopologyGenLoadCountKind::kSink;
  std::string clock_name;
  std::string clock_net_name;
  std::string sink_domain;
  std::string stage;
  SchemaWriter* reporter = nullptr;
};

struct TopologyGenConfig
{
  BiPartitionConfig partition_config;
  std::optional<unsigned> target_depth = std::nullopt;
};

class TopologyGen
{
 public:
  using Input = TopologyGenInput;
  using Config = TopologyGenConfig;
  using LoadCountKind = TopologyGenLoadCountKind;

  TopologyGen() = delete;
  ~TopologyGen() = default;

  static auto build(const std::vector<Pin*>& loads, const Input& input, const Config& config) -> Tree;

 private:
  struct BuildCursor
  {
    std::size_t node_id = 0;
    int depth = 0;
  };

  static auto reportLoadDistribution(SchemaWriter* reporter, const std::vector<Pin*>& loads, LoadCountKind load_count_kind,
                                     int32_t dbu_per_um) -> void;
  static auto reportRootToLeafLengths(SchemaWriter* reporter, const Tree& tree, int32_t dbu_per_um) -> void;
  static auto calcMaxDepth(std::size_t load_count) -> unsigned;
  static auto calcLeafCount(std::size_t load_count) -> std::size_t;
  static auto buildWithConfig(const std::vector<Pin*>& loads, const Input& input, const Config& config) -> Tree;
  static auto buildFullTree(Tree& tree, const BuildCursor& cursor, int height) -> void;
  static auto embedPositions(Tree& tree, std::size_t node, const std::vector<Pin*>& loads, std::size_t leaf_need,
                             const BiPartitionConfig& config) -> void;
  static auto balanceTopology(Tree& tree, int min_x, int min_y, int max_x, int max_y, double topology_tolerance) -> void;
};

}  // namespace icts
