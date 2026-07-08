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
 * @file TopologyRealTechClusterAssertions.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief GTest adapters for Topology real-tech cluster validation.
 */

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "TopologyRealTechScenario.hh"
#include "synthesis/topology/Topology.hh"

namespace icts {
class Inst;
class Pin;
}  // namespace icts

namespace icts_test::synthesis_realtech_smoke {
namespace {

auto EmitValidationFailures(const TopologyValidationResult& validation) -> void
{
  for (const auto& failure_message : validation.failure_messages) {
    ADD_FAILURE() << failure_message;
  }
}

}  // namespace

auto AssertClusteredSinkConnectivity(const std::vector<icts::Pin*>& sinks, const std::unordered_set<icts::Inst*>& cluster_buffer_insts)
    -> void
{
  EmitValidationFailures(ValidateClusteredSinkConnectivity(sinks, cluster_buffer_insts));
}

auto AssertClusterBufferMastersFollowLeafSemantics(const icts::Topology::Build& result, const std::string& min_cluster_master) -> void
{
  EmitValidationFailures(ValidateClusterBufferMastersFollowLeafSemantics(result, min_cluster_master));
}

}  // namespace icts_test::synthesis_realtech_smoke
