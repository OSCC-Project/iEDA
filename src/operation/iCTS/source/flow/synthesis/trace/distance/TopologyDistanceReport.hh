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
 * @file TopologyDistanceReport.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Emits Topology-specific structured report sections.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "synthesis/topology/Topology.hh"

namespace icts {

class SchemaWriter;
}  // namespace icts

namespace icts::topology {

struct ClusterLeafDistanceReportInput
{
  std::string log_file;
  std::int32_t dbu_per_um = 0;
  SchemaWriter* reporter = nullptr;
  const Topology::Build* build = nullptr;
};

auto EmitClusterLeafDistanceTables(const ClusterLeafDistanceReportInput& input) -> std::optional<Topology::ClusterLeafDistanceSummary>;

}  // namespace icts::topology
