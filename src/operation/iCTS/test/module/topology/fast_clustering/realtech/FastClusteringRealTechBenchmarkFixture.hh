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
 * @file FastClusteringRealTechBenchmarkFixture.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Shared declarations for the fast-clustering real-tech benchmark.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "module/topology/clustering/Clustering.hh"
#include "module/topology/config/TopologyConfig.hh"

namespace icts_test::fast_clustering::realtech {

inline constexpr std::string_view kBenchmarkRoot = "/nfs/share/home/huangzhipeng/code-new/ecc-benchmark/runs/20260422_125008";
inline constexpr std::string_view kIcs55Workspace = "/home/liweiguo/project/ecc-tools-dev/scripts/design/ics55_dev";
inline constexpr std::string_view kCtsConfigPath
    = "/home/liweiguo/project/ecc-tools-dev/scripts/design/ics55_dev/iEDA_config/cts_default_config.json";
inline constexpr std::string_view kDefaultSdcPath = "/home/liweiguo/project/ecc-tools-dev/scripts/design/ics55_dev/default.sdc";
inline constexpr std::size_t kRequiredCaseCount = 20;
inline constexpr std::string_view kClusterSvgDirName = "cluster_svgs";

struct BenchmarkCase
{
  std::size_t index = 0;
  std::string case_name;
  std::string top_module;
  std::filesystem::path case_dir;
  std::filesystem::path def_path;
  std::filesystem::path verilog_path;
};

struct TechAssets
{
  std::filesystem::path pdk_root;
  std::filesystem::path cts_config_path;
  std::filesystem::path sdc_path;
  std::filesystem::path tech_lef_path;
  std::vector<std::filesystem::path> lef_paths;
  std::vector<std::filesystem::path> lib_paths;
};

struct LoadedCase
{
  bool ok = false;
  std::string error;
  int dbu_per_micron = 0;
  std::size_t inst_count = 0;
  std::size_t net_count = 0;
  std::size_t clock_count = 0;
  std::string clock_name;
  std::string clock_net_name;
  std::string clock_selection_reason;
  std::vector<icts::Pin*> loads;
  std::size_t load_count = 0;
  int span_diameter = 0;
};

struct ClockNetCandidate
{
  std::string net_name;
  std::string reason;
  std::size_t load_pin_count = 0;
  std::size_t inst_load_count = 0;
  std::size_t clock_pin_count = 0;
  bool idb_clock = false;
  bool name_clock_like = false;
  long long score = 0;
};

struct ResultMetrics
{
  std::string algorithm;
  double runtime_ms = 0.0;
  bool legal = false;
  std::size_t expected_load_count = 0;
  std::size_t load_count = 0;
  std::size_t missing_load_count = 0;
  std::size_t cluster_count = 0;
  std::size_t singleton_count = 0;
  std::size_t max_fanout = 0;
  int max_diameter = 0;
  std::size_t fanout_violations = 0;
  std::size_t diameter_violations = 0;
  std::size_t cap_violations = 0;
  std::size_t route_failures = 0;
  double total_score = 0.0;
  double total_wirelength = 0.0;
  double total_routing_cap_proxy = 0.0;
  double avg_routing_cap_proxy = 0.0;
  double routing_cap_proxy_variance = 0.0;
  double routing_cap_proxy_stddev = 0.0;
};

struct CaseResult
{
  BenchmarkCase benchmark_case;
  LoadedCase loaded;
  ResultMetrics fast;
  std::string cluster_svg;
};

struct ClusterRunResult
{
  ResultMetrics metrics;
  icts::ClusterOutput result;
};

struct ClusterSvgArtifacts
{
  std::unordered_map<const icts::Pin*, std::size_t> cluster_map;
  std::vector<icts::Point<int>> centers;
  std::vector<std::size_t> cluster_sizes;
};

auto ContainsClockToken(std::string_view name) -> bool;
auto DiscoverBenchmarkCases() -> std::vector<BenchmarkCase>;
auto ResolveTechAssets() -> TechAssets;
auto ValidateTechAssets(const TechAssets& assets, std::string& error) -> bool;
auto LoadBenchmarkCase(const BenchmarkCase& benchmark_case, const TechAssets& assets, const std::filesystem::path& output_dir)
    -> LoadedCase;
auto CalcClusterDiameter(const std::vector<icts::Pin*>& loads) -> int;
auto BuildBenchmarkConfig(const std::vector<icts::Pin*>& loads) -> icts::ClusterConfig;
auto EvaluateResult(const std::string& algorithm, const icts::ClusterOutput& result, const icts::ClusterConfig& config,
                    std::size_t expected_load_count, double runtime_ms) -> ResultMetrics;
auto WriteCaseClusterSvg(const std::filesystem::path& svg_dir, const BenchmarkCase& benchmark_case, const std::vector<icts::Pin*>& loads,
                         const icts::ClusterOutput& fast_result, std::string& error) -> std::filesystem::path;
auto BuildCasesCsv(const std::vector<CaseResult>& results) -> std::string;
auto BuildMetricsCsv(const std::vector<CaseResult>& results) -> std::string;
auto BuildVisualizationCsv(const std::vector<CaseResult>& results) -> std::string;
auto BuildInventoryReport(const std::vector<BenchmarkCase>& cases, const TechAssets& assets) -> std::string;
auto BuildLoadedCaseReport(const BenchmarkCase& benchmark_case, const LoadedCase& loaded) -> std::string;
auto SumFastRoutingCapProxyVariance(const std::vector<CaseResult>& results) -> double;
auto BuildSummaryReport(const std::vector<CaseResult>& results, double fast_runtime_ms, double fast_score) -> std::string;

template <typename Runner>
auto RunAndMeasure(const std::string& algorithm, const std::vector<icts::Pin*>& loads, const icts::ClusterConfig& config, Runner runner)
    -> ClusterRunResult
{
  const auto start = std::chrono::steady_clock::now();
  auto result = runner(loads, config);
  const auto finish = std::chrono::steady_clock::now();
  const auto runtime_ms = std::chrono::duration<double, std::milli>(finish - start).count();
  return ClusterRunResult{.metrics = EvaluateResult(algorithm, result, config, loads.size(), runtime_ms), .result = std::move(result)};
}

}  // namespace icts_test::fast_clustering::realtech
