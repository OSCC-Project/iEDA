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
 * @file TopologyBuildTrace.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Records clock-tree synthesis metrics and transfers temporary object ownership.
 */

#pragma once

#include "synthesis/htree/HTree.hh"
#include "synthesis/topology/Topology.hh"
#include "synthesis/topology/trunk/SourceTrunkSegment.hh"

namespace icts::topology {

struct SourceTrunkBuild;

auto RecordSinkHtreeBuild(Topology::Build& build, HTree::Build htree_build) -> void;
auto RecordTopSegmentBuild(SourceTrunkBuild& build, SourceTrunkSegment::Build& segment_build) -> void;
auto RecordTopHtreeBuild(SourceTrunkBuild& build, HTree::Build htree_build) -> void;

}  // namespace icts::topology
