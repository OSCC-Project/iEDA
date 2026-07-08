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
 * @file FastClusteringRealTechBenchmarkTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech CTS clustering benchmark entry point for fast clustering.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "Clustering.hh"
#include "FastClusteringRealTechBenchmarkFixture.hh"
#include "common/dataset/TestDataset.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/logging/ScopedLogFile.hh"
#include "module/topology/fast_clustering/FastClustering.hh"
#include "utils/logger/Schema.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::fast_clustering::realtech {
namespace {

using common::io::PrepareCleanOutputDir;
using common::io::ResolveOutputDir;
using common::io::WriteRawTextLog;
using common::logging::ScopedLogFile;

TEST(FastClusteringRealTechBenchmarkTest, BenchmarkTwentyPlacementCases)
{
  const auto output_dir = PrepareCleanOutputDir(ResolveOutputDir() / "fast_clustering" / "realtech_benchmark" / "current_run");
  const auto cts_log_path = output_dir / "cts.log";
  ScopedLogFile scoped_log(cts_log_path, "Fast Clustering RealTech Benchmark");

  const auto cases = DiscoverBenchmarkCases();
  const auto assets = ResolveTechAssets();
  common::io::EmitInfoReport(InfoReport{.title = "CTS Clustering Benchmark Inventory", .content = BuildInventoryReport(cases, assets)});

  if (cases.empty()) {
    GTEST_SKIP() << "benchmark root unavailable: " << kBenchmarkRoot;
  }
  ASSERT_EQ(cases.size(), kRequiredCaseCount);

  std::string tech_error;
  ASSERT_TRUE(ValidateTechAssets(assets, tech_error)) << tech_error;

  std::vector<CaseResult> results;
  results.reserve(cases.size());
  double fast_runtime_ms = 0.0;
  double fast_score = 0.0;
  const auto svg_dir = output_dir / std::string(kClusterSvgDirName);

  for (const auto& benchmark_case : cases) {
    auto loaded = LoadBenchmarkCase(benchmark_case, assets, output_dir);
    ASSERT_TRUE(loaded.ok) << benchmark_case.case_name << ": " << loaded.error;
    common::io::EmitInfoReport(
        InfoReport{.title = "CTS Clustering Case Statistics", .content = BuildLoadedCaseReport(benchmark_case, loaded)});

    auto config = BuildBenchmarkConfig(loaded.loads);
    auto fast_run = RunAndMeasure("fast", loaded.loads, config,
                                  [](const std::vector<icts::Pin*>& loads, const auto& run_config) -> icts::ClusterOutput {
                                    return icts::FastClustering::runDefault(loads, run_config);
                                  });

    fast_runtime_ms += fast_run.metrics.runtime_ms;
    fast_score += fast_run.metrics.total_score;

    std::string svg_error;
    const auto svg_path = WriteCaseClusterSvg(svg_dir, benchmark_case, loaded.loads, fast_run.result, svg_error);
    EXPECT_FALSE(svg_path.empty()) << benchmark_case.case_name << ": " << svg_error;
    std::string cluster_svg;
    if (!svg_path.empty()) {
      cluster_svg = (std::filesystem::path(std::string(kClusterSvgDirName)) / svg_path.filename()).string();
      icts::EmitArtifact(icts_test::runtime::CurrentRuntime().reporter, "CTS clustering structure svg", svg_path);
    }

    loaded.loads.clear();
    results.push_back(CaseResult{.benchmark_case = benchmark_case,
                                 .loaded = std::move(loaded),
                                 .fast = std::move(fast_run.metrics),
                                 .cluster_svg = std::move(cluster_svg)});
  }

  auto summary = BuildSummaryReport(results, fast_runtime_ms, fast_score);
  summary += "visualization_svg_dir=" + std::string(kClusterSvgDirName) + "\n";
  summary += "visualization_svg_count=" + std::to_string(results.size()) + "\n";
  WriteRawTextLog(output_dir / "cts_clustering_cases.csv", BuildCasesCsv(results));
  WriteRawTextLog(output_dir / "cts_clustering_metrics.csv", BuildMetricsCsv(results));
  WriteRawTextLog(output_dir / "cts_clustering_visualizations.csv", BuildVisualizationCsv(results));
  WriteRawTextLog(output_dir / "report.log", summary);
  common::io::EmitInfoReport(InfoReport{.title = "CTS Clustering Benchmark Summary", .content = summary});

  for (const auto& result : results) {
    EXPECT_TRUE(result.fast.legal) << result.benchmark_case.case_name << " fast illegal";
    EXPECT_EQ(result.fast.missing_load_count, 0U) << result.benchmark_case.case_name;
    EXPECT_EQ(result.fast.fanout_violations, 0U) << result.benchmark_case.case_name;
    EXPECT_EQ(result.fast.diameter_violations, 0U) << result.benchmark_case.case_name;
    EXPECT_EQ(result.fast.cap_violations, 0U) << result.benchmark_case.case_name;
    EXPECT_EQ(result.fast.route_failures, 0U) << result.benchmark_case.case_name;
  }
}

}  // namespace
}  // namespace icts_test::fast_clustering::realtech
