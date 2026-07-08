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
 * @file TopologyBuildTrace.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Implements clock-tree synthesis metrics and temporary ownership transfers.
 */

#include "synthesis/trace/topology_build/TopologyBuildTrace.hh"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "synthesis/htree/HTree.hh"
#include "synthesis/topology/trunk/SourceTrunk.hh"

namespace icts::topology {
namespace {

template <typename T>
auto absorbOwnedObjects(std::vector<std::unique_ptr<T>>& target, std::vector<std::unique_ptr<T>>& source) -> void
{
  target.reserve(target.size() + source.size());
  for (auto& object : source) {
    target.push_back(std::move(object));
  }
  source.clear();
}

auto absorbHtreeInsertedObjects(Topology::Build& build, HTree::Output& htree_output) -> void
{
  build.output.inserted_inst_levels.insert(build.output.inserted_inst_levels.end(), htree_output.inserted_inst_levels.begin(),
                                           htree_output.inserted_inst_levels.end());
  build.output.inserted_net_levels.insert(build.output.inserted_net_levels.end(), htree_output.inserted_net_levels.begin(),
                                          htree_output.inserted_net_levels.end());
  absorbOwnedObjects(build.output.inserted_insts, htree_output.inserted_insts);
  absorbOwnedObjects(build.output.inserted_pins, htree_output.inserted_pins);
  absorbOwnedObjects(build.output.inserted_nets, htree_output.inserted_nets);
}

auto absorbHtreeInsertedObjects(SourceTrunkBuild& build, HTree::Output& htree_output) -> void
{
  build.output.inserted_inst_levels.insert(build.output.inserted_inst_levels.end(), htree_output.inserted_inst_levels.begin(),
                                           htree_output.inserted_inst_levels.end());
  build.output.inserted_net_levels.insert(build.output.inserted_net_levels.end(), htree_output.inserted_net_levels.begin(),
                                          htree_output.inserted_net_levels.end());
  absorbOwnedObjects(build.output.inserted_insts, htree_output.inserted_insts);
  absorbOwnedObjects(build.output.inserted_pins, htree_output.inserted_pins);
  absorbOwnedObjects(build.output.inserted_nets, htree_output.inserted_nets);
}

auto absorbSegmentInsertedObjects(SourceTrunkBuild& build, SourceTrunkSegment::Build& segment_build) -> void
{
  build.output.inserted_inst_levels.insert(build.output.inserted_inst_levels.end(), segment_build.output.inserted_inst_levels.begin(),
                                           segment_build.output.inserted_inst_levels.end());
  build.output.inserted_net_levels.insert(build.output.inserted_net_levels.end(), segment_build.output.inserted_net_levels.begin(),
                                          segment_build.output.inserted_net_levels.end());
  absorbOwnedObjects(build.output.inserted_insts, segment_build.output.inserted_insts);
  absorbOwnedObjects(build.output.inserted_pins, segment_build.output.inserted_pins);
  absorbOwnedObjects(build.output.inserted_nets, segment_build.output.inserted_nets);
}

auto selectedLeafTopologyLevel(const HTree::Output& htree_output, const std::optional<unsigned>& selected_depth) -> int
{
  if (selected_depth.has_value()) {
    return static_cast<int>(*selected_depth);
  }
  return htree_output.levels.empty() ? 0 : static_cast<int>(htree_output.levels.size());
}

auto recordClusterSinkNetLevels(Topology::Build& build) -> void
{
  const int topology_level = selectedLeafTopologyLevel(build.output.htree_output, build.summary.selected_htree_depth);
  std::size_t index_in_level = 0U;
  for (const auto& cluster_buffer : build.output.cluster_buffers) {
    if (cluster_buffer.sink_net == nullptr) {
      continue;
    }
    build.output.inserted_net_levels.push_back(HTree::InsertedNetLevel{
        .net = cluster_buffer.sink_net,
        .topology_level = topology_level,
        .index_in_level = index_in_level++,
    });
  }
}

}  // namespace

auto RecordSinkHtreeBuild(Topology::Build& build, HTree::Build htree_build) -> void
{
  build.summary.selected_htree_level_count = htree_build.output.levels.size();
  build.summary.selected_htree_depth = htree_build.summary.selected_depth;
  build.summary.htree_inserted_buffer_count = htree_build.output.inserted_insts.size();
  build.summary.htree_inserted_net_count = htree_build.output.inserted_nets.size();
  build.output.htree_output = std::move(htree_build.output);
  recordClusterSinkNetLevels(build);
  absorbHtreeInsertedObjects(build, build.output.htree_output);
}

auto RecordTopSegmentBuild(SourceTrunkBuild& build, SourceTrunkSegment::Build& segment_build) -> void
{
  build.summary.used_boundary_relaxation = segment_build.summary.used_boundary_relaxation;
  build.summary.selected_level_count = segment_build.output.inserted_insts.size();
  build.summary.inserted_buffer_count = segment_build.output.inserted_insts.size();
  build.summary.inserted_net_count = segment_build.output.inserted_nets.size();
  absorbSegmentInsertedObjects(build, segment_build);
  build.summary.success = true;
}

auto RecordTopHtreeBuild(SourceTrunkBuild& build, HTree::Build htree_build) -> void
{
  build.summary.selected_depth = htree_build.summary.selected_depth;
  build.summary.selected_level_count = htree_build.output.levels.size();
  build.summary.used_boundary_relaxation = htree_build.summary.used_boundary_relaxation;
  build.summary.inserted_buffer_count = htree_build.output.inserted_insts.size();
  build.summary.inserted_net_count = htree_build.output.inserted_nets.size();
  build.output.htree_output = std::move(htree_build.output);
  absorbHtreeInsertedObjects(build, build.output.htree_output);
  build.summary.success = true;
}

}  // namespace icts::topology
