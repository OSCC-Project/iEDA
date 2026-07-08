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
 * @file FastClusteringRealTechBenchmarkArtifacts.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief CSV, report, and SVG artifact writers for the fast-clustering real-tech benchmark.
 */

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "FastClusteringRealTechBenchmarkFixture.hh"
#include "common/clustering/artifact/ClusterArtifactWriter.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/visualization/TestVisualization.hh"

namespace icts {
class Pin;
struct ClusterOutput;
}  // namespace icts

namespace icts_test::fast_clustering::realtech {
namespace {
using common::io::SanitizeOutputName;

auto CsvEscape(const std::string& text) -> std::string
{
  if (text.find_first_of(",\"\n") == std::string::npos) {
    return text;
  }

  std::string escaped = "\"";
  for (const auto ch : text) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += '"';
  return escaped;
}

auto FormatCaseIndex(std::size_t index) -> std::string
{
  std::ostringstream stream;
  stream << std::setw(2) << std::setfill('0') << index;
  return stream.str();
}

auto BuildClusterSvgName(const BenchmarkCase& benchmark_case) -> std::string
{
  return "case_" + FormatCaseIndex(benchmark_case.index) + "_" + SanitizeOutputName(benchmark_case.case_name) + "_clusters.svg";
}

}  // namespace

auto WriteCaseClusterSvg(const std::filesystem::path& svg_dir, const BenchmarkCase& benchmark_case, const std::vector<icts::Pin*>& loads,
                         const icts::ClusterOutput& fast_result, std::string& error) -> std::filesystem::path
{
  std::error_code error_code;
  std::filesystem::create_directories(svg_dir, error_code);
  if (error_code) {
    error = "failed to create cluster svg directory: " + error_code.message();
    return {};
  }

  ClusterSvgArtifacts fast_artifacts;
  if (!common::clustering::BuildClusterArtifacts(fast_result, loads, fast_artifacts.cluster_map, fast_artifacts.centers,
                                                 fast_artifacts.cluster_sizes, error)) {
    error = "fast artifacts: " + error;
    return {};
  }

  auto svg_path = svg_dir / BuildClusterSvgName(benchmark_case);
  const bool wrote_svg
      = common::visualization::WriteClusterSvg(svg_path.string(), loads, fast_artifacts.cluster_map, fast_artifacts.centers);
  if (!wrote_svg) {
    error = "failed to write fast cluster svg: " + svg_path.string();
    return {};
  }
  return svg_path;
}

auto BuildCasesCsv(const std::vector<CaseResult>& results) -> std::string
{
  std::ostringstream stream;
  stream << "index,case,top,def,verilog,dbu_per_micron,inst_count,net_count,clock_count,clock_name,clock_net,"
            "clock_selection_reason,load_count,span_diameter\n";
  for (const auto& result : results) {
    stream << result.benchmark_case.index << "," << CsvEscape(result.benchmark_case.case_name) << ","
           << CsvEscape(result.benchmark_case.top_module) << "," << CsvEscape(result.benchmark_case.def_path.string()) << ","
           << CsvEscape(result.benchmark_case.verilog_path.string()) << "," << result.loaded.dbu_per_micron << ","
           << result.loaded.inst_count << "," << result.loaded.net_count << "," << result.loaded.clock_count << ","
           << CsvEscape(result.loaded.clock_name) << "," << CsvEscape(result.loaded.clock_net_name) << ","
           << CsvEscape(result.loaded.clock_selection_reason) << "," << result.loaded.load_count << "," << result.loaded.span_diameter
           << "\n";
  }
  return stream.str();
}

namespace {

auto AppendMetricsCsvRow(std::ostringstream& stream, const BenchmarkCase& benchmark_case, const ResultMetrics& metrics) -> void
{
  stream << benchmark_case.index << "," << CsvEscape(benchmark_case.case_name) << "," << metrics.algorithm << "," << std::fixed
         << std::setprecision(3) << metrics.runtime_ms << "," << std::setprecision(6) << metrics.total_score << "," << metrics.legal << ","
         << metrics.expected_load_count << "," << metrics.load_count << "," << metrics.missing_load_count << "," << metrics.cluster_count
         << "," << metrics.singleton_count << "," << metrics.max_fanout << "," << metrics.max_diameter << "," << metrics.fanout_violations
         << "," << metrics.diameter_violations << "," << metrics.cap_violations << "," << metrics.route_failures << ","
         << metrics.total_wirelength << "," << metrics.total_routing_cap_proxy << "," << metrics.avg_routing_cap_proxy << ","
         << metrics.routing_cap_proxy_variance << "," << metrics.routing_cap_proxy_stddev << "\n";
}

}  // namespace

auto BuildMetricsCsv(const std::vector<CaseResult>& results) -> std::string
{
  std::ostringstream stream;
  stream << "index,case,algorithm,runtime_ms,total_score,legal,expected_load_count,load_count,missing_load_count,"
            "cluster_count,singleton_count,max_fanout,max_diameter,"
            "fanout_violations,diameter_violations,cap_violations,route_failures,total_wirelength,"
            "total_routing_cap_proxy,avg_routing_cap_proxy,routing_cap_proxy_variance,routing_cap_proxy_stddev\n";
  for (const auto& result : results) {
    AppendMetricsCsvRow(stream, result.benchmark_case, result.fast);
  }
  return stream.str();
}

auto BuildVisualizationCsv(const std::vector<CaseResult>& results) -> std::string
{
  std::ostringstream stream;
  stream << "index,case,cluster_svg\n";
  for (const auto& result : results) {
    stream << result.benchmark_case.index << "," << CsvEscape(result.benchmark_case.case_name) << "," << CsvEscape(result.cluster_svg)
           << "\n";
  }
  return stream.str();
}

auto BuildInventoryReport(const std::vector<BenchmarkCase>& cases, const TechAssets& assets) -> std::string
{
  std::ostringstream stream;
  stream << "benchmark_root=" << kBenchmarkRoot << "\n";
  stream << "case_count=" << cases.size() << "\n";
  stream << "pdk_root=" << assets.pdk_root.string() << "\n";
  stream << "cts_config=" << assets.cts_config_path.string() << "\n";
  stream << "sdc=" << assets.sdc_path.string() << "\n";
  stream << "tech_lef=" << assets.tech_lef_path.string() << "\n";
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& benchmark_case = cases.at(index);
    stream << "case_" << (index + 1U) << "=" << benchmark_case.case_name << ",top=" << benchmark_case.top_module
           << ",def=" << benchmark_case.def_path.string() << ",verilog=" << benchmark_case.verilog_path.string() << "\n";
  }
  return stream.str();
}

auto BuildLoadedCaseReport(const BenchmarkCase& benchmark_case, const LoadedCase& loaded) -> std::string
{
  std::ostringstream stream;
  stream << "index=" << benchmark_case.index << "\n";
  stream << "case=" << benchmark_case.case_name << "\n";
  stream << "top=" << benchmark_case.top_module << "\n";
  stream << "dbu_per_micron=" << loaded.dbu_per_micron << "\n";
  stream << "inst_count=" << loaded.inst_count << "\n";
  stream << "net_count=" << loaded.net_count << "\n";
  stream << "clock_count=" << loaded.clock_count << "\n";
  stream << "clock_name=" << loaded.clock_name << "\n";
  stream << "clock_net=" << loaded.clock_net_name << "\n";
  stream << "clock_selection_reason=" << loaded.clock_selection_reason << "\n";
  stream << "load_count=" << loaded.load_count << "\n";
  stream << "span_diameter=" << loaded.span_diameter << "\n";
  return stream.str();
}

auto SumFastRoutingCapProxyVariance(const std::vector<CaseResult>& results) -> double
{
  double total = 0.0;
  for (const auto& result : results) {
    total += result.fast.routing_cap_proxy_variance;
  }
  return total;
}

auto BuildSummaryReport(const std::vector<CaseResult>& results, double fast_runtime_ms, double fast_score) -> std::string
{
  std::size_t legal_cases = 0;
  std::size_t missing_load_count = 0;
  std::size_t fanout_violations = 0;
  std::size_t diameter_violations = 0;
  std::size_t cap_violations = 0;
  std::size_t route_failures = 0;
  const auto fast_routing_cap_variance = SumFastRoutingCapProxyVariance(results);
  for (const auto& result : results) {
    if (result.fast.legal) {
      ++legal_cases;
    }
    missing_load_count += result.fast.missing_load_count;
    fanout_violations += result.fast.fanout_violations;
    diameter_violations += result.fast.diameter_violations;
    cap_violations += result.fast.cap_violations;
    route_failures += result.fast.route_failures;
  }

  std::ostringstream stream;
  stream << "case_count=" << results.size() << "\n";
  stream << "legal_case_count=" << legal_cases << "\n";
  stream << "fast_total_runtime_ms=" << fast_runtime_ms << "\n";
  stream << "fast_total_score=" << fast_score << "\n";
  stream << "fast_routing_cap_proxy_variance_sum=" << fast_routing_cap_variance << "\n";
  stream << "fast_missing_load_count=" << missing_load_count << "\n";
  stream << "fast_fanout_violations=" << fanout_violations << "\n";
  stream << "fast_diameter_violations=" << diameter_violations << "\n";
  stream << "fast_cap_violations=" << cap_violations << "\n";
  stream << "fast_route_failures=" << route_failures << "\n";
  return stream.str();
}

}  // namespace icts_test::fast_clustering::realtech
