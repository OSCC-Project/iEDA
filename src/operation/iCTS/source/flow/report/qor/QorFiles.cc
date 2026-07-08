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
 * @file QorFiles.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-27
 * @brief CTS statistics report writer implementation.
 */

#include "report/qor/QorFiles.hh"

#include <glog/logging.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "Log.hh"
#include "Qor.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"

namespace icts {
namespace {

auto formatReportNumber(double value) -> std::string
{
  return logformat::FormatFixed(value, 3);
}

auto formatCurrentWallTime() -> std::string
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t time_value = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &time_value);
#else
  localtime_r(&time_value, &local_tm);
#endif

  std::ostringstream stream;
  stream << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
  return stream.str();
}

auto buildWirelengthReportRows(const Qor& statistics) -> TableRows
{
  return {
      {"Top", formatReportNumber(statistics.top_wirelength_um), formatReportNumber(statistics.hpwl_top_wirelength_um)},
      {"Trunk", formatReportNumber(statistics.trunk_wirelength_um), formatReportNumber(statistics.hpwl_trunk_wirelength_um)},
      {"Leaf", formatReportNumber(statistics.leaf_wirelength_um), formatReportNumber(statistics.hpwl_leaf_wirelength_um)},
      {"Total", formatReportNumber(statistics.total_wirelength_um), formatReportNumber(statistics.hpwl_total_wirelength_um)},
      {"Max Net Length", formatReportNumber(statistics.max_net_wirelength_um), formatReportNumber(statistics.hpwl_max_net_wirelength_um)},
  };
}

auto buildCellStatsReportRows(const Qor& statistics) -> TableRows
{
  TableRows rows;
  for (const auto& [cell_type, stats] : statistics.cell_stats) {
    rows.push_back({
        cell_type,
        std::to_string(stats.count),
        formatReportNumber(stats.total_area_um2),
        formatReportNumber(stats.total_cap_pf),
    });
  }
  if (rows.empty()) {
    rows.push_back({"none", "0", "0.000", "0.000"});
  }
  return rows;
}

auto buildLibCellDistReportRows(const Qor& statistics) -> TableRows
{
  TableRows rows;
  for (const auto& [cell_master, distribution] : statistics.lib_cell_dist) {
    rows.push_back({
        cell_master,
        distribution.cell_type,
        std::to_string(distribution.count),
        formatReportNumber(distribution.total_area_um2),
    });
  }
  if (rows.empty()) {
    rows.push_back({"none", "none", "0", "0.000"});
  }
  return rows;
}

auto writeReportFile(const std::filesystem::path& path, const std::string& title, const std::vector<std::string>& headers,
                     const TableRows& rows) -> bool
{
  std::ofstream stream(path, std::ios::out | std::ios::trunc);
  if (!stream.is_open()) {
    LOG_WARNING << "QorFiles: failed to open statistics report " << path.string();
    return false;
  }

  stream << "Generate the report at " << formatCurrentWallTime() << '\n';
  stream << logformat::MakeTitledTable(title, headers, rows);
  if (!rows.empty()) {
    stream << '\n';
  }
  return true;
}

}  // namespace

auto QorFiles::writeReports(const std::filesystem::path& statistics_dir, const Qor& statistics) -> bool
{
  if (statistics_dir.empty()) {
    LOG_WARNING << "QorFiles: statistics report directory is empty.";
    return false;
  }

  std::error_code error_code;
  std::filesystem::create_directories(statistics_dir, error_code);
  if (error_code) {
    LOG_WARNING << "QorFiles: failed to create statistics directory " << statistics_dir.string();
    return false;
  }

  bool success = true;
  success &= writeReportFile(statistics_dir / "wirelength.rpt", "Wirelength Statistics",
                             {"Metric", "Routed Wirelength (um)", "HPWL Wirelength (um)"}, buildWirelengthReportRows(statistics));
  success &= writeReportFile(statistics_dir / "cell_stats.rpt", "Cell Stats", {"Cell Type", "Count", "Total Area (um^2)", "Total Cap (pF)"},
                             buildCellStatsReportRows(statistics));
  success &= writeReportFile(statistics_dir / "lib_cell_dist.rpt", "Library Cell Distribution",
                             {"Cell Master", "Cell Type", "Count", "Total Area (um^2)"}, buildLibCellDistReportRows(statistics));
  return success;
}

auto QorFiles::emitLogTables(SchemaWriter& reporter, const Qor& statistics) -> void
{
  reporter.emitSection("### Statistics Reports");
  EmitTable(reporter, "CTS Statistics Wirelength Report", {"Metric", "Routed Wirelength (um)", "HPWL Wirelength (um)"},
            buildWirelengthReportRows(statistics));
  EmitTable(reporter, "CTS Statistics Cell Stats Report", {"Cell Type", "Count", "Total Area (um^2)", "Total Cap (pF)"},
            buildCellStatsReportRows(statistics));
  EmitTable(reporter, "CTS Statistics Library Cell Distribution Report", {"Cell Master", "Cell Type", "Count", "Total Area (um^2)"},
            buildLibCellDistReportRows(statistics));
}

}  // namespace icts
