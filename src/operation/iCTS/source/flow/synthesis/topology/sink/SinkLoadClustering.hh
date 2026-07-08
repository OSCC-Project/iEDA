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
 * @file SinkLoadClustering.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Prepares clustered or direct HTree sink loads for Topology.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "synthesis/topology/Topology.hh"

namespace icts {
class Wrapper;
class Pin;
}  // namespace icts

namespace icts::topology {

struct SinkTreeLoadPreparationPolicy
{
  bool enable_sink_clustering = false;
  std::size_t max_fanout = 0U;
  double max_cap_pf = 0.0;
  ClockRouteSegmentRc clock_route_segment_rc;
  const std::vector<std::string>* buffer_cell_masters = nullptr;
};

struct SinkTreeLoadPreparationInput
{
  Topology::Build* build = nullptr;
  const std::vector<Pin*>* root_loads = nullptr;
  Wrapper* wrapper = nullptr;
  std::string object_name_prefix;
  SinkTreeLoadPreparationPolicy policy;
};

struct SinkTreeLoadPreparation
{
  bool success = false;
  std::vector<Pin*> htree_sinks;
};

auto PrepareSinkTreeLoads(const SinkTreeLoadPreparationInput& input) -> SinkTreeLoadPreparation;

}  // namespace icts::topology
