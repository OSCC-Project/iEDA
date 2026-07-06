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
#include "report/ReportWriter.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <omp.h>

#include "FormatUtils.hh"
#include "PathUtils.hh"
#include "libfort/fort.hpp"
#include "log/Log.hh"

namespace ircx {
namespace compare_spef {
namespace {

struct ErrorStats
{
  static constexpr int kMinBin = -32;
  static constexpr int kMaxBin = 32;
  static constexpr int kBinStep = 2;

  double mean_value = 0.0;
  double m2 = 0.0;
  double min = 0.0;
  double max = 0.0;
  std::size_t count = 0;
  std::map<int, std::size_t> bins;

  ErrorStats()
  {
    for (int bin = kMinBin; bin <= kMaxBin; bin += kBinStep) {
      bins[bin] = 0;
    }
  }

  void add(double value)
  {
    if (count == 0) {
      min = value;
      max = value;
    } else {
      min = std::min(min, value);
      max = std::max(max, value);
    }

    ++count;
    const double delta = value - mean_value;
    mean_value += delta / static_cast<double>(count);
    const double delta2 = value - mean_value;
    m2 += delta * delta2;

    int bin = static_cast<int>(std::round(value / static_cast<double>(kBinStep))) * kBinStep;
    bin = std::clamp(bin, kMinBin, kMaxBin);
    bins[bin]++;
  }

  auto mean() const -> double
  {
    return count == 0 ? 0.0 : mean_value;
  }

  auto standardDeviation() const -> double
  {
    if (count == 0) {
      return 0.0;
    }

    return std::sqrt(m2 / static_cast<double>(count));
  }
};

struct SummaryErrors
{
  ErrorStats tcap;
  ErrorStats gcap;
  ErrorStats ccap;
  ErrorStats p2p;
};

auto openReport(const std::filesystem::path& path) -> std::ofstream
{
  std::ofstream ofs(path);
  ofs << std::fixed << std::setprecision(3);
  return ofs;
}

void writeDouble(std::ostream& os, double value)
{
  if (std::isfinite(value)) {
    os << value;
  } else {
    os << "NA";
  }
}

void writePercent(std::ostream& os, const std::optional<double>& value)
{
  if (value.has_value()) {
    os << (*value * 100.0);
  } else {
    os << "NA";
  }
}

template <typename Row>
void collectPercentErrors(const std::vector<Row>& rows, ErrorStats& stats)
{
  for (const auto& row : rows) {
    if (row.relative_delta.has_value()) {
      stats.add(*row.relative_delta * 100.0);
    }
  }
}

auto collectSummaryErrors(const Result& result) -> SummaryErrors
{
  SummaryErrors errors;
  collectPercentErrors(result.tcap_rows, errors.tcap);
  collectPercentErrors(result.gcap_rows, errors.gcap);
  collectPercentErrors(result.ccap_rows, errors.ccap);
  collectPercentErrors(result.p2p_rows, errors.p2p);
  return errors;
}

class SummaryTableBuilder
{
 public:
  auto makeOverviewTable(const Config& config, const SummaryErrors& errors) const -> fort::char_table
  {
    auto table = makePlainTable();
    table << fort::header << "Metric" << "Mean Error" << "Std Error" << "Threshold" << fort::endr;
    table << "Total cap (C)" << format::percent(errors.tcap.mean())
          << format::percent(errors.tcap.standardDeviation())
          << "abs = " + format::fixed(config.tcap_threshold) + "fF" << fort::endr;
    table << "Ground cap (GC)" << format::percent(errors.gcap.mean())
          << format::percent(errors.gcap.standardDeviation())
          << "abs = " + format::fixed(config.ccap_abs_threshold) + "fF" << fort::endr;
    table << "Coupling cap (CC)" << format::percent(errors.ccap.mean())
          << format::percent(errors.ccap.standardDeviation())
          << "abs = " + format::fixed(config.ccap_abs_threshold) + "fF, rel = " + format::fixed(config.ccap_rel_threshold)
          << fort::endr;
    table << "Pin-Pin res (P2P)" << format::percent(errors.p2p.mean())
          << format::percent(errors.p2p.standardDeviation())
          << "abs = " + format::fixed(config.res_threshold) + "Ohm" << fort::endr;
    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(3).set_cell_text_align(fort::text_align::left);
    return table;
  }

  auto makeDistributionTable(const std::string& title, const std::string& threshold_label, double threshold, const std::string& count_label,
                             const ErrorStats& errors) const -> fort::char_table
  {
    auto table = makePlainTable();
    table << fort::header << title + " Distribution" << "Value" << fort::endr;
    table << threshold_label << format::fixed(threshold) << fort::endr;
    table << "Min Error" << format::percent(errors.count == 0 ? 0.0 : errors.min) << fort::endr;
    table << "Max Error" << format::percent(errors.count == 0 ? 0.0 : errors.max) << fort::endr;
    table << "Mean Error" << format::percent(errors.mean()) << fort::endr;
    table << "Standard dev" << format::percent(errors.standardDeviation()) << fort::endr;
    table << count_label << errors.count << fort::endr;
    table << fort::separator;
    table << fort::header << "Error Bin" << "Count" << fort::endr;
    for (const auto& [bin, count] : errors.bins) {
      table << std::to_string(bin) + "%" << count << fort::endr;
    }
    table.column(0).set_cell_text_align(fort::text_align::left);
    return table;
  }

 private:
  auto makePlainTable() const -> fort::char_table
  {
    fort::char_table table;
    table.set_border_style(FT_PLAIN_STYLE);
    table.set_cell_text_align(fort::text_align::right);
    table.row(0).set_cell_text_align(fort::text_align::center);
    return table;
  }
};

auto writeTcapReport(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  const auto report_path = output_dir / "tcap.rpt";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }
  ofs << config.test_file << '\t' << config.reference_file << "\t%diff\tNetname\n";
  for (const auto& row : result.tcap_rows) {
    ofs << row.test << '\t' << row.reference << '\t';
    writePercent(ofs, row.relative_delta);
    ofs << '\t' << row.net << '\n';
  }
  return true;
}

auto writeGcapReport(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  const auto report_path = output_dir / "gcap.rpt";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }
  ofs << config.test_file << '\t' << config.reference_file << "\t%diff\tNetname\n";
  for (const auto& row : result.gcap_rows) {
    ofs << row.test << '\t' << row.reference << '\t';
    writePercent(ofs, row.relative_delta);
    ofs << '\t' << row.net << '\n';
  }
  return true;
}

auto writeCcapReport(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  const auto report_path = output_dir / "ccap.rpt";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }
  ofs << config.test_file << '\t' << config.reference_file << "\t%diff\tVictim\tAggressor\tTCAP Victim\n";
  for (const auto& row : result.ccap_rows) {
    ofs << row.test << '\t' << row.reference << '\t';
    writePercent(ofs, row.relative_delta);
    ofs << '\t' << row.victim << '\t' << row.aggressor << '\t' << row.reference_victim_total_cap << '\n';
  }
  return true;
}

auto writeP2PReport(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  const auto report_path = output_dir / "p2p.rpt";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }
  ofs << config.test_file << '\t' << config.reference_file << "\t%diff\tNetname\tPin1\tPin2\n";
  for (const auto& row : result.p2p_rows) {
    writeDouble(ofs, row.test);
    ofs << '\t';
    writeDouble(ofs, row.reference);
    ofs << '\t';
    writePercent(ofs, row.relative_delta);
    ofs << '\t' << row.net << '\t' << row.from_pin << '\t' << row.to_pin << '\n';
  }
  return true;
}

auto writeMismatchedNets(const std::filesystem::path& output_dir, const Result& result) -> bool
{
  const auto report_path = output_dir / "nets.mismatched";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }
  ofs << "Total# of nets in reference: " << result.summary.reference_net_count << '\n';
  ofs << "Total# of nets in test: " << result.summary.test_net_count << "\n\n";
  ofs << "# of missing nets: " << result.summary.reference_only_net_count << '\n';
  for (const auto& net : result.reference_only_nets) {
    ofs << "\tMIT: " << net << "\t0.000\n";
  }
  ofs << "\n# of extra nets: " << result.summary.test_only_net_count << '\n';
  for (const auto& net : result.test_only_nets) {
    ofs << "\tMIG: " << net << "\t0.000\n";
  }
  return true;
}

auto writeMismatchedCouplings(const std::filesystem::path& output_dir, const Result& result) -> bool
{
  const auto report_path = output_dir / "coupling_caps.mismatched";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }

  ofs << "Total# of coupling caps in reference: " << result.summary.reference_coupling_count << '\n';
  ofs << "Total# of coupling caps in test: " << result.summary.test_coupling_count << "\n\n";
  ofs << "# of missing coupling caps: " << result.summary.reference_only_coupling_count << '\n';
  for (const auto& mismatch : result.reference_only_couplings) {
    ofs << "\tMIT: " << mismatch.report_nets.first << '\t' << mismatch.report_nets.second << '\t' << mismatch.capacitance << '\n';
  }
  ofs << "\n# of extra coupling caps: " << result.summary.test_only_coupling_count << '\n';
  for (const auto& mismatch : result.test_only_couplings) {
    ofs << "\tMIG: " << mismatch.report_nets.first << '\t' << mismatch.report_nets.second << '\t' << mismatch.capacitance << '\n';
  }
  return true;
}

void writeSummaryHeader(std::ostream& ofs, const Config& config)
{
  ofs << "\n ##########################################################\n";
  ofs << " #                                                        #\n";
  ofs << " #               compare_spef Utility                    #\n";
  ofs << " #                                                        #\n";
  ofs << " ##########################################################\n\n";
  ofs << "=====================================================================\n\n";
  ofs << "                 compare_spef Utility\n\n";
  ofs << "\t(IN)           MY RC FILE : " << config.test_file << '\n';
  ofs << "\t(IN)         GOLD RC FILE : " << config.reference_file << '\n';
  ofs << "\t(OUT)      SUMMARY REPORT : summary.rpt\n";
  ofs << "\t(OUT)    TOTAL CAP REPORT : tcap.rpt\n";
  ofs << "\t(OUT)   GROUND CAP REPORT : gcap.rpt\n";
  ofs << "\t(OUT) COUPLING CAP REPORT : ccap.rpt\n";
  ofs << "\t(OUT)      RES P2P REPORT : p2p.rpt\n\n";
  ofs << "=====================================================================\n\n\n";
}

auto writeSummaryReport(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  const auto report_path = output_dir / "summary.rpt";
  auto ofs = openReport(report_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open report " << report_path;
    return false;
  }

  const auto errors = collectSummaryErrors(result);
  const SummaryTableBuilder builder;
  writeSummaryHeader(ofs, config);
  ofs << "RC Correlation Overview\n";
  ofs << builder.makeOverviewTable(config, errors).to_string() << '\n';
  ofs << builder.makeDistributionTable("TCAP", "TCAP threshold", config.tcap_threshold, "Number of matched nets", errors.tcap).to_string()
      << '\n';
  ofs << builder.makeDistributionTable("GCAP", "GCAP threshold", config.ccap_abs_threshold, "Number of matched nets", errors.gcap)
             .to_string()
      << '\n';
  ofs << builder.makeDistributionTable("CCAP", "CCAP threshold", config.ccap_abs_threshold, "Number of matched net pairs", errors.ccap)
             .to_string()
      << '\n';
  ofs << builder.makeDistributionTable("P2P", "RES threshold", config.res_threshold, "Number of matched pin pairs", errors.p2p).to_string();
  return true;
}

}  // namespace

namespace {

using ReportTask = bool (*)(const std::filesystem::path&, const Config&, const Result&);
using SimpleReportTask = bool (*)(const std::filesystem::path&, const Result&);

auto writeTcapTask(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  return writeTcapReport(output_dir, config, result);
}

auto writeCcapTask(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  return writeCcapReport(output_dir, config, result);
}

auto writeGcapTask(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  return writeGcapReport(output_dir, config, result);
}

auto writeP2PTask(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  return writeP2PReport(output_dir, config, result);
}

auto writeSummaryTask(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  return writeSummaryReport(output_dir, config, result);
}

auto writeNetsTask(const std::filesystem::path& output_dir, const Result& result) -> bool
{
  return writeMismatchedNets(output_dir, result);
}

auto writeCouplingsTask(const std::filesystem::path& output_dir, const Result& result) -> bool
{
  return writeMismatchedCouplings(output_dir, result);
}

auto writeReports(const std::filesystem::path& output_dir, const Config& config, const Result& result) -> bool
{
  constexpr std::array<ReportTask, 5> report_tasks = {writeTcapTask, writeGcapTask, writeCcapTask, writeP2PTask, writeSummaryTask};
  constexpr std::array<SimpleReportTask, 2> simple_report_tasks = {writeNetsTask, writeCouplingsTask};

  std::array<bool, report_tasks.size() + simple_report_tasks.size()> ok;
  ok.fill(false);

  const int thread_count = std::min<int>(static_cast<int>(config.cores), ok.size());
#pragma omp parallel for schedule(static) num_threads(thread_count)
  for (std::size_t index = 0; index < ok.size(); ++index) {
    if (index < report_tasks.size()) {
      ok[index] = report_tasks[index](output_dir, config, result);
    } else {
      ok[index] = simple_report_tasks[index - report_tasks.size()](output_dir, result);
    }
  }

  return std::all_of(ok.begin(), ok.end(), [](bool value) { return value; });
}

}  // namespace

ReportWriter::ReportWriter(const Config& config) : _config(config)
{
}

auto ReportWriter::write(const Result& result) const -> bool
{
  if (!path::ensure_dir(_config.output_dir, "output_dir")) {
    return false;
  }

  const std::filesystem::path output_dir(_config.output_dir);
  const bool ok = writeReports(output_dir, _config, result);
  if (!ok) {
    LOG_ERROR << "compare_spef failed: cannot write one or more reports under " << _config.output_dir;
  }
  return ok;
}

}  // namespace compare_spef
}  // namespace ircx
