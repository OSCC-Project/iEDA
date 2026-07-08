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
 * @file ReportExport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS report result export path policy implementation.
 */

#include "report/export/ReportExport.hh"

#include "config/Config.hh"

namespace icts {
namespace {

auto resolveReportRootDir(const Config& config, const std::string& save_dir) -> std::filesystem::path
{
  if (!save_dir.empty()) {
    return std::filesystem::path(save_dir);
  }
  return std::filesystem::path(config.get_work_dir());
}

auto resolveVisualizationDir(const Config& config, const std::string& save_dir, const std::filesystem::path& report_root_dir)
    -> std::filesystem::path
{
  if (!save_dir.empty()) {
    return report_root_dir / "visualization";
  }
  if (!config.get_visualization_dir().empty()) {
    return std::filesystem::path(config.get_visualization_dir());
  }
  return report_root_dir / "visualization";
}

auto resolveStatisticsDir(const Config& config, const std::string& save_dir, const std::filesystem::path& report_root_dir)
    -> std::filesystem::path
{
  if (!save_dir.empty()) {
    return report_root_dir / "statistics";
  }
  if (!config.get_statistics_dir().empty()) {
    return std::filesystem::path(config.get_statistics_dir());
  }
  return report_root_dir / "statistics";
}

}  // namespace

auto ReportExport::resolvePaths(const Config& config, const std::string& save_dir) -> ReportExportPaths
{
  const auto report_root_dir = resolveReportRootDir(config, save_dir);
  return ReportExportPaths{
      .report_root_dir = report_root_dir,
      .visualization_dir = resolveVisualizationDir(config, save_dir, report_root_dir),
      .statistics_dir = resolveStatisticsDir(config, save_dir, report_root_dir),
  };
}

}  // namespace icts
