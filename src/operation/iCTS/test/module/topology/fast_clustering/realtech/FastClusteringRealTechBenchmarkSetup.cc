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
 * @file FastClusteringRealTechBenchmarkSetup.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Design loading and clock-net selection for the fast-clustering real-tech benchmark.
 */

#include <algorithm>
#include <compare>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "FastClusteringRealTechBenchmarkFixture.hh"
#include "IdbDesign.h"
#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "common/io/TestArtifactIO.hh"
#include "database/config/Config.hh"
#include "database/design/Clock.hh"
#include "database/design/Design.hh"
#include "database/io/Wrapper.hh"
#include "dm_config.h"
#include "idm.h"
#include "setup/clock_data/ClockDataRead.hh"

namespace icts_test::fast_clustering::realtech {
namespace {
using common::io::SanitizeOutputName;

auto LoadTechnologyOnce(const TechAssets& assets, std::string& error) -> bool
{
  static bool loaded = false;
  if (loaded) {
    return true;
  }

  if (!dmInst->readLef(std::vector<std::string>{assets.tech_lef_path.string()}, true)) {
    error = "readLef(tech) failed";
    return false;
  }

  std::vector<std::string> lef_paths;
  lef_paths.reserve(assets.lef_paths.size());
  for (const auto& lef_path : assets.lef_paths) {
    lef_paths.push_back(lef_path.string());
  }
  if (!dmInst->readLef(lef_paths, false)) {
    error = "readLef(cells) failed";
    return false;
  }

  loaded = true;
  return true;
}

auto SetupDataManagerPaths(const TechAssets& assets, const BenchmarkCase& benchmark_case, const std::filesystem::path& output_dir) -> void
{
  auto& dm_config = dmInst->get_config();
  dm_config.set_tech_lef_path(assets.tech_lef_path.string());

  std::vector<std::string> lef_paths = {assets.tech_lef_path.string()};
  for (const auto& lef_path : assets.lef_paths) {
    lef_paths.push_back(lef_path.string());
  }
  dm_config.set_lef_paths(lef_paths);
  dm_config.set_def_path(benchmark_case.def_path.string());

  std::vector<std::string> lib_paths;
  lib_paths.reserve(assets.lib_paths.size());
  for (const auto& lib_path : assets.lib_paths) {
    lib_paths.push_back(lib_path.string());
  }
  dm_config.set_lib_paths(lib_paths);
  dm_config.set_sdc_path(assets.sdc_path.string());
  dm_config.set_output_path((output_dir / "dm_output" / SanitizeOutputName(benchmark_case.case_name)).string());
}

auto BuildClockNetCandidate(idb::IdbNet* idb_net) -> ClockNetCandidate
{
  ClockNetCandidate candidate;
  if (idb_net == nullptr) {
    return candidate;
  }

  candidate.net_name = idb_net->get_net_name();
  candidate.idb_clock = idb_net->is_clock();
  candidate.name_clock_like = ContainsClockToken(candidate.net_name);
  const auto idb_load_pins = idb_net->get_load_pins();
  candidate.load_pin_count = idb_load_pins.size();
  for (auto* idb_pin : idb_load_pins) {
    if (idb_pin == nullptr || idb_pin->get_instance() == nullptr) {
      continue;
    }
    ++candidate.inst_load_count;
    if (idb_pin->is_flip_flop_clk()) {
      ++candidate.clock_pin_count;
    }
  }

  candidate.score = static_cast<long long>(candidate.inst_load_count);
  if (candidate.name_clock_like) {
    candidate.score += 10'000'000'000LL;
  }
  if (candidate.clock_pin_count > 0U) {
    candidate.score += 100'000'000'000LL + static_cast<long long>(candidate.clock_pin_count) * 1'000'000LL;
  }
  if (candidate.idb_clock) {
    candidate.score += 1'000'000'000'000LL;
  }

  std::ostringstream reason_stream;
  reason_stream << "idb_clock=" << candidate.idb_clock << ",name_clock_like=" << candidate.name_clock_like
                << ",clock_pin_count=" << candidate.clock_pin_count << ",inst_load_count=" << candidate.inst_load_count;
  candidate.reason = reason_stream.str();
  return candidate;
}

auto SelectClockNetCandidate(idb::IdbDesign* idb_design) -> std::optional<ClockNetCandidate>
{
  if (idb_design == nullptr || idb_design->get_net_list() == nullptr) {
    return std::nullopt;
  }

  std::optional<ClockNetCandidate> best_candidate = std::nullopt;
  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    if (idb_net == nullptr || idb_net->is_pdn()) {
      continue;
    }
    auto candidate = BuildClockNetCandidate(idb_net);
    if (candidate.inst_load_count <= 1U) {
      continue;
    }
    if (!best_candidate.has_value() || candidate.score > best_candidate->score
        || (candidate.score == best_candidate->score && candidate.net_name < best_candidate->net_name)) {
      best_candidate = std::move(candidate);
    }
  }
  return best_candidate;
}

}  // namespace

auto LoadBenchmarkCase(const BenchmarkCase& benchmark_case, const TechAssets& assets, const std::filesystem::path& output_dir) -> LoadedCase
{
  LoadedCase loaded;
  icts_test::runtime::CurrentRuntime().config.reset();
  icts_test::runtime::CurrentRuntime().config.init(assets.cts_config_path.string());
  icts_test::runtime::CurrentRuntime().config.set_work_dir(
      (output_dir / "cts_workspace" / SanitizeOutputName(benchmark_case.case_name)).string());

  std::string tech_error;
  if (!LoadTechnologyOnce(assets, tech_error)) {
    loaded.error = tech_error;
    return loaded;
  }

  if (!dmInst->readVerilog(benchmark_case.verilog_path.string(), benchmark_case.top_module)) {
    loaded.error = "readVerilog failed";
    return loaded;
  }
  if (!dmInst->readDef(benchmark_case.def_path.string())) {
    loaded.error = "readDef failed";
    return loaded;
  }

  SetupDataManagerPaths(assets, benchmark_case, output_dir);
  auto* idb_builder = dmInst->get_idb_builder();
  if (idb_builder == nullptr) {
    loaded.error = "idb builder is null after placement load";
    return loaded;
  }

  icts_test::runtime::CurrentRuntime().design.reset();
  auto* idb_design = dmInst->get_idb_design();
  if (idb_design == nullptr) {
    loaded.error = "iDB design is null after load";
    return loaded;
  }
  if (idb_design->get_design_name() != benchmark_case.top_module) {
    loaded.error = "DEF/Verilog design mismatch: def=" + idb_design->get_design_name() + ", verilog=" + benchmark_case.top_module;
    return loaded;
  }

  icts_test::runtime::CurrentRuntime().wrapper.reset();
  icts_test::runtime::CurrentRuntime().wrapper.init(idb_builder);
  auto& runtime = icts_test::runtime::CurrentRuntime();
  if (!icts::ClockDataRead::read(icts::ClockDataReadInput{
          .config = &runtime.config,
          .design = &runtime.design,
          .wrapper = &runtime.wrapper,
          .reporter = &runtime.reporter,
      })) {
    loaded.error = "readClockData failed for SDC-declared clocks";
    return loaded;
  }

  loaded.dbu_per_micron = icts_test::runtime::CurrentRuntime().wrapper.queryDbUnit();
  if (loaded.dbu_per_micron <= 0) {
    loaded.error = "DBU-per-micron unavailable after DEF load";
    return loaded;
  }
  loaded.inst_count = idb_design->get_instance_list() == nullptr ? 0U : idb_design->get_instance_list()->get_instance_list().size();
  loaded.net_count = idb_design->get_net_list() == nullptr ? 0U : idb_design->get_net_list()->get_net_list().size();
  const auto clocks = icts_test::runtime::CurrentRuntime().design.get_clocks();
  loaded.clock_count = clocks.size();
  for (auto* clock : clocks) {
    if (clock == nullptr || clock->get_loads().size() <= loaded.loads.size()) {
      continue;
    }
    loaded.clock_name = clock->get_clock_name();
    loaded.clock_net_name = clock->get_clock_net_name();
    loaded.loads = clock->get_loads();
    loaded.clock_selection_reason = "sdc_declared_clock";
  }
  if (loaded.loads.empty()) {
    loaded.error = "no clock loads extracted";
    return loaded;
  }

  loaded.load_count = loaded.loads.size();
  loaded.span_diameter = CalcClusterDiameter(loaded.loads);
  loaded.ok = true;
  return loaded;
}

}  // namespace icts_test::fast_clustering::realtech
