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
 * @file HTreeArtifactWriter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-14
 * @brief SVG artifact emission helpers for HTree tests.
 */

#include "flow/synthesis/htree/HTreeArtifactWriter.hh"

#include <cmath>
#include <string>
#include <vector>

#include "common/io/TestArtifactIO.hh"
#include "common/visualization/TestVisualization.hh"
#include "flow/synthesis/htree/HTreeSvgRenderer.hh"
#include "synthesis/htree/HTree.hh"
#include "utils/logger/Schema.hh"

namespace icts_test::htree {

auto PrepareHTreeArtifactPaths(const std::string& case_name) -> HTreeArtifactPaths
{
  HTreeArtifactPaths paths;
  paths.output_dir
      = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "flow" / "htree" / common::io::SanitizeOutputName(case_name));
  if (paths.output_dir.empty()) {
    return paths;
  }

  paths.cts_log = paths.output_dir / "cts.log";
  paths.topology_svg = paths.output_dir / "topology.svg";
  paths.materialized_svg = paths.output_dir / "materialized_htree.svg";
  paths.pareto_svg = paths.output_dir / "pareto_delay_power.svg";
  paths.report_log = paths.output_dir / "report.log";
  return paths;
}

auto WriteHTreeArtifacts(const HTreeArtifactPaths& paths, const std::string& scenario_name, const std::string& input_summary,
                         const std::vector<icts::Pin*>& loads, const icts::htree::DiagnosticBuild& result) -> bool
{
  if (paths.output_dir.empty()) {
    return false;
  }

  const bool wrote_topology = common::visualization::WriteTopologySvg(paths.topology_svg.string(), result.output.topology, loads);
  const bool wrote_materialized = WriteMaterializedSvg(paths.materialized_svg, loads, result);
  const bool wrote_pareto = WriteDelayPowerParetoSvg(paths.pareto_svg, result);
  const bool wrote_report = common::io::WriteTextLog(paths.report_log, BuildReport(scenario_name, input_summary, paths, loads, result));
  auto& reporter = icts_test::runtime::CurrentRuntime().reporter;

  if (wrote_topology) {
    icts::EmitArtifact(reporter, "HTree topology svg", paths.topology_svg);
  }
  if (wrote_materialized) {
    icts::EmitArtifact(reporter, "HTree materialized svg", paths.materialized_svg);
  }
  if (wrote_pareto) {
    icts::EmitArtifact(reporter, "HTree delay-power pareto svg", paths.pareto_svg);
  }
  if (wrote_report) {
    icts::EmitArtifact(reporter, "HTree report", paths.report_log);
  }

  return wrote_topology && wrote_materialized && wrote_pareto && wrote_report;
}

}  // namespace icts_test::htree
