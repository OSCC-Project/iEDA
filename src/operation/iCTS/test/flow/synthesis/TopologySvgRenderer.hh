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
 * @file TopologySvgRenderer.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Private SVG rendering contracts for topology visualization tests.
 */

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Inst.hh"
#include "Pin.hh"
#include "Point.hh"
#include "flow/synthesis/topology/Topology.hh"

namespace icts_test::synthesis {

constexpr double kTopologyVisualizationBufferHalfSize = 6.0;

struct BufferMasterSummary
{
  std::string cell_master;
  std::size_t count = 0;
  int drive_strength = -1;
};

struct BufferRenderStyle
{
  std::string fill_color;
  std::string stroke_color;
  double half_size = kTopologyVisualizationBufferHalfSize;
};

struct LineSegment
{
  icts::Point<int> start = icts::Point<int>(-1, -1);
  icts::Point<int> end = icts::Point<int>(-1, -1);
};

auto HasValidLocation(const icts::Point<int>& location) -> bool;
auto FindRenderableLocation(const icts::Pin* pin) -> icts::Point<int>;
auto CollectExtraPoints(const icts::Topology::Build& result) -> std::vector<icts::Point<int>>;
auto CollectBufferMasterSummaries(const std::vector<std::unique_ptr<icts::Inst>>& inserted_insts) -> std::vector<BufferMasterSummary>;
auto BuildBufferRenderStyles(const std::vector<BufferMasterSummary>& summaries) -> std::unordered_map<std::string, BufferRenderStyle>;
auto CollectSinkLevelSegments(const icts::Topology::Build& result) -> std::vector<LineSegment>;
auto WriteSynthesisSvg(const std::filesystem::path& path, const std::vector<icts::Pin*>& original_sinks,
                       const icts::Topology::Build& result) -> bool;

}  // namespace icts_test::synthesis
