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
 * @file SvgVisualization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief CTS clock-tree SVG visualization generation.
 */

#pragma once

#include <filesystem>

namespace icts {
class ClockLayout;
class Config;
class Design;
class SchemaWriter;
class Wrapper;

}  // namespace icts

namespace icts::visualization {

struct SvgVisualizationInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  SchemaWriter* reporter = nullptr;
  std::filesystem::path visualization_dir;
  const ClockLayout* clock_layout = nullptr;
};

struct SvgVisualizationSummary
{
  bool success = false;
};

auto EmitSvgVisualizations(const SvgVisualizationInput& input) -> SvgVisualizationSummary;

}  // namespace icts::visualization
