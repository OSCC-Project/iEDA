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
 * @file Visualization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS clock-tree visualization report output implementation.
 */

#include "report/visualization/Visualization.hh"

#include <glog/logging.h>

#include <ostream>

#include "Log.hh"
#include "report/visualization/gds/GdsVisualization.hh"
#include "report/visualization/svg/SvgVisualization.hh"

namespace icts {

auto Visualization::emit(const VisualizationInput& input, const VisualizationConfig& config) -> VisualizationSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "Visualization requires config.";
  LOG_FATAL_IF(input.design == nullptr) << "Visualization requires design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Visualization requires wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Visualization requires reporter.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "Visualization requires clock layout.";
  const auto svg_summary
      = config.emit_svg
            ? visualization::EmitSvgVisualizations(visualization::SvgVisualizationInput{.config = input.config,
                                                                                        .design = input.design,
                                                                                        .wrapper = input.wrapper,
                                                                                        .reporter = input.reporter,
                                                                                        .visualization_dir = input.visualization_dir,
                                                                                        .clock_layout = input.clock_layout})
            : visualization::SvgVisualizationSummary{.success = true};
  const auto gds_summary
      = config.emit_gds
            ? visualization::EmitGdsVisualizations(visualization::GdsVisualizationInput{.config = input.config,
                                                                                        .design = input.design,
                                                                                        .wrapper = input.wrapper,
                                                                                        .reporter = input.reporter,
                                                                                        .visualization_dir = input.visualization_dir,
                                                                                        .clock_layout = input.clock_layout})
            : visualization::GdsVisualizationSummary{.success = true};
  return VisualizationSummary{
      .svg_success = svg_summary.success,
      .gds_success = gds_summary.success,
      .success = svg_summary.success && gds_summary.success,
  };
}

}  // namespace icts
