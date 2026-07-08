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
 * @file RealTechAssetLoader.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Asset probing and environment bootstrap for real-tech tests.
 */

#include "common/realtech/asset/RealTechAssetLoader.hh"

#include <glog/logging.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "Flow.hh"
#include "Log.hh"
#include "common/CTSTestRuntime.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/config/Config.hh"
#include "database/io/Wrapper.hh"
#include "dm_config.h"
#include "idm.h"
#include "setup/clock_data/ClockDataRead.hh"

namespace icts_test::common::realtech::asset {
namespace {

struct RealTechAssets
{
  std::filesystem::path workspace_path;
  std::filesystem::path pdk_root_path;
  std::filesystem::path flow_script_path;
  std::filesystem::path cts_config_path;
  std::filesystem::path tech_lef_path;
  std::vector<std::filesystem::path> lef_paths;
  std::filesystem::path def_path;
  std::filesystem::path verilog_path;
  std::string top_module;
  std::vector<std::filesystem::path> lib_paths;
  std::filesystem::path sdc_path;
  std::filesystem::path output_dir;
};

constexpr std::string_view kRealTechWorkspaceEnv = "ICTS_REALTECH_WORKSPACE";
constexpr std::string_view kRealTechPdkEnv = "ICTS_REALTECH_PDK_DIR";
constexpr std::string_view kLegacyPdkEnv = "PDK_DIR";
constexpr std::array<std::string_view, 2> kDefaultWorkspaceRelPaths = {
    "scripts/design/ics55_dev",
    "scripts/design/ics55_gcd",
};
constexpr std::array<std::string_view, 2> kFlowScriptRelPaths = {
    "script/iCTS_script/run_iCTS_dev.tcl",
    "script/iCTS_script/run_iCTS.tcl",
};
constexpr std::string_view kRunScriptRelPath = "run_iEDA.sh";
constexpr std::string_view kCtsConfigRelPath = "iEDA_config/cts_default_config.json";
constexpr std::array<std::string_view, 5> kRealTechDefRelPaths = {
    "result/bp_be_top_place.def.gz", "result/bp_be_top_place.def", "result/arm9_place.def.gz",
    "result/arm9_place.def",         "result/iPL_result.def",
};
constexpr std::array<std::string_view, 5> kRealTechVerilogRelPaths = {
    "result/bp_be_top_place.v", "result/arm9_place.v", "result/iCTS_result.v", "result/iPL_result.v", "result/verilog/gcd_nl.v",
};

auto TrimAscii(const std::string& text) -> std::string
{
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1U);
}

auto StripWrappingQuotes(std::string text) -> std::string
{
  text = TrimAscii(text);
  if (text.size() >= 2U) {
    const char first = text.front();
    const char last = text.back();
    if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
      text = text.substr(1U, text.size() - 2U);
    }
  }
  return TrimAscii(text);
}

auto ReadEnvPath(std::string_view env_name) -> std::filesystem::path
{
  const auto env_key = std::string(env_name);
  const char* env_value = std::getenv(env_key.c_str());
  return (env_value != nullptr && *env_value != '\0') ? std::filesystem::path(env_value) : std::filesystem::path{};
}

auto ResolveRepoRootFromSource() -> std::filesystem::path
{
  auto current = std::filesystem::absolute(std::filesystem::path(__FILE__)).parent_path();
  while (!current.empty()) {
    if (std::filesystem::exists(current / "scripts") && std::filesystem::exists(current / "src")) {
      return current;
    }
    const auto parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }
  return {};
}

auto ResolveExistingPath(const std::filesystem::path& base_path, const std::array<std::string_view, 2>& relative_candidates)
    -> std::filesystem::path
{
  for (const auto& relative_path : relative_candidates) {
    const auto candidate = base_path / relative_path;
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return base_path / relative_candidates.front();
}

template <std::size_t N>
auto ResolveExistingPath(const std::filesystem::path& base_path, const std::array<std::string_view, N>& relative_candidates)
    -> std::filesystem::path
{
  for (const auto& relative_path : relative_candidates) {
    const auto candidate = base_path / relative_path;
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return base_path / relative_candidates.front();
}

auto TryParseShellExport(const std::filesystem::path& script_path, std::string_view variable_name) -> std::filesystem::path
{
  std::ifstream input_stream(script_path);
  if (!input_stream) {
    return {};
  }

  const std::string expected_prefix = "export " + std::string(variable_name) + "=";
  std::string line;
  while (std::getline(input_stream, line)) {
    auto trimmed_line = TrimAscii(line);
    if (!trimmed_line.starts_with(expected_prefix)) {
      continue;
    }
    return StripWrappingQuotes(trimmed_line.substr(expected_prefix.size()));
  }
  return {};
}

auto TryParseTclSet(const std::filesystem::path& script_path, std::string_view variable_name) -> std::filesystem::path
{
  std::ifstream input_stream(script_path);
  if (!input_stream) {
    return {};
  }

  std::string line;
  while (std::getline(input_stream, line)) {
    auto trimmed_line = TrimAscii(line);
    if (!trimmed_line.starts_with("set ")) {
      continue;
    }

    std::istringstream line_stream(trimmed_line);
    std::string set_keyword;
    std::string parsed_name;
    line_stream >> set_keyword >> parsed_name;
    if (parsed_name != variable_name) {
      continue;
    }

    std::string value;
    std::getline(line_stream, value);
    return StripWrappingQuotes(value);
  }
  return {};
}

auto TryParseVerilogTopModule(const std::filesystem::path& verilog_path) -> std::string
{
  std::ifstream input_stream(verilog_path);
  if (!input_stream) {
    return {};
  }

  std::string line;
  while (std::getline(input_stream, line)) {
    const auto trimmed_line = TrimAscii(line);
    if (!trimmed_line.starts_with("module ")) {
      continue;
    }

    std::istringstream line_stream(trimmed_line);
    std::string module_keyword;
    std::string module_name;
    line_stream >> module_keyword >> module_name;
    if (module_keyword != "module" || module_name.empty()) {
      continue;
    }

    while (!module_name.empty() && (module_name.back() == '(' || module_name.back() == ';')) {
      module_name.pop_back();
    }
    return module_name;
  }
  return {};
}

auto ResolvePdkRootPath(const std::filesystem::path& workspace_path) -> std::filesystem::path
{
  const auto absolutize_to_workspace = [&workspace_path](std::filesystem::path candidate_path) -> std::filesystem::path {
    if (!candidate_path.empty() && candidate_path.is_relative()) {
      candidate_path = std::filesystem::absolute(workspace_path / candidate_path);
    }
    return candidate_path;
  };

  if (const auto explicit_pdk_root = ReadEnvPath(kRealTechPdkEnv); !explicit_pdk_root.empty()) {
    return explicit_pdk_root;
  }
  if (const auto legacy_pdk_root = ReadEnvPath(kLegacyPdkEnv); !legacy_pdk_root.empty()) {
    return legacy_pdk_root;
  }

  if (const auto run_script_pdk = TryParseShellExport(workspace_path / kRunScriptRelPath, "PDK_DIR"); !run_script_pdk.empty()) {
    return absolutize_to_workspace(run_script_pdk);
  }

  if (const auto flow_script = ResolveExistingPath(workspace_path, kFlowScriptRelPaths); std::filesystem::exists(flow_script)) {
    return absolutize_to_workspace(TryParseTclSet(flow_script, "PDK_DIR"));
  }

  return {};
}

auto BuildAssetsFromWorkspace(const std::filesystem::path& workspace_path) -> std::optional<RealTechAssets>
{
  if (workspace_path.empty()) {
    return std::nullopt;
  }

  RealTechAssets assets;
  assets.workspace_path = workspace_path;
  assets.pdk_root_path = ResolvePdkRootPath(workspace_path);
  assets.flow_script_path = ResolveExistingPath(workspace_path, kFlowScriptRelPaths);
  assets.cts_config_path = workspace_path / kCtsConfigRelPath;
  assets.tech_lef_path = assets.pdk_root_path / "prtech/techLEF/N551P6M_ecos.lef";
  assets.lef_paths = {
      assets.pdk_root_path / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR_ecos.lef",
      assets.pdk_root_path / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL_ecos.lef",
  };
  assets.def_path = ResolveExistingPath(workspace_path, kRealTechDefRelPaths);
  assets.verilog_path = ResolveExistingPath(workspace_path, kRealTechVerilogRelPaths);
  if (!assets.verilog_path.empty() && std::filesystem::exists(assets.verilog_path)) {
    assets.top_module = TryParseVerilogTopModule(assets.verilog_path);
  }
  assets.lib_paths = {
      assets.pdk_root_path / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib",
      assets.pdk_root_path / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_ss_rcworst_1p08_125_nldm.lib",
  };
  assets.sdc_path = workspace_path / "default.sdc";
  assets.output_dir = workspace_path / "result";
  return assets;
}

auto CollectCandidateAssets() -> std::vector<RealTechAssets>
{
  std::vector<RealTechAssets> candidates;
  std::set<std::string> seen_workspaces;

  auto append_workspace = [&](const std::filesystem::path& workspace_path) -> void {
    if (workspace_path.empty()) {
      return;
    }
    const auto normalized_workspace = std::filesystem::absolute(workspace_path).lexically_normal();
    if (!seen_workspaces.insert(normalized_workspace.string()).second) {
      return;
    }

    if (auto assets = BuildAssetsFromWorkspace(normalized_workspace); assets.has_value()) {
      candidates.push_back(std::move(*assets));
    }
  };

  append_workspace(ReadEnvPath(kRealTechWorkspaceEnv));

  if (const auto repo_root = ResolveRepoRootFromSource(); !repo_root.empty()) {
    for (const auto& relative_workspace : kDefaultWorkspaceRelPaths) {
      append_workspace(repo_root / relative_workspace);
    }
  }

  return candidates;
}

auto ValidateAssets(const RealTechAssets& assets, std::string& error) -> bool
{
  std::vector<std::string> missing_entries;
  auto check_file = [&missing_entries](const std::string& label, const std::filesystem::path& path) -> void {
    if (path.empty() || !std::filesystem::exists(path)) {
      missing_entries.emplace_back(label + "=" + path.string());
    }
  };

  check_file("workspace", assets.workspace_path);
  check_file("pdk_root", assets.pdk_root_path);
  check_file("flow_script", assets.flow_script_path);
  check_file("cts_config", assets.cts_config_path);
  check_file("tech_lef", assets.tech_lef_path);
  check_file("def", assets.def_path);
  if (!assets.verilog_path.empty()) {
    check_file("verilog", assets.verilog_path);
  }
  check_file("sdc", assets.sdc_path);
  if (assets.lef_paths.empty()) {
    missing_entries.emplace_back("lef_paths=<empty>");
  } else {
    for (const auto& lef_path : assets.lef_paths) {
      check_file("lef", lef_path);
    }
  }
  if (assets.lib_paths.empty()) {
    missing_entries.emplace_back("lib_path=<empty>");
  } else {
    for (const auto& lib_path : assets.lib_paths) {
      check_file("lib", lib_path);
    }
  }

  if (missing_entries.empty()) {
    return true;
  }

  std::ostringstream stream;
  stream << "missing assets: ";
  for (std::size_t index = 0; index < missing_entries.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << missing_entries.at(index);
  }
  error = stream.str();
  return false;
}

auto LoadRealTechAssets(const RealTechAssets& assets, std::string& error) -> bool
{
  if (assets.cts_config_path.empty() || !std::filesystem::exists(assets.cts_config_path)) {
    error = "cts config is missing: " + assets.cts_config_path.string();
    return false;
  }
  icts_test::runtime::CurrentRuntime().config.init(assets.cts_config_path.string());

  const std::vector<std::string> tech_lef_paths = {assets.tech_lef_path.string()};
  if (!dmInst->readLef(tech_lef_paths, true)) {
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

  if (!assets.verilog_path.empty() && std::filesystem::exists(assets.verilog_path)) {
    if (assets.top_module.empty()) {
      error = "cannot resolve top module from verilog: " + assets.verilog_path.string();
      return false;
    }
    if (!dmInst->readVerilog(assets.verilog_path.string(), assets.top_module)) {
      error = "readVerilog failed";
      return false;
    }
  }

  if (!dmInst->readDef(assets.def_path.string())) {
    error = "readDef failed";
    return false;
  }

  auto& dm_config = dmInst->get_config();
  dm_config.set_tech_lef_path(assets.tech_lef_path.string());

  std::vector<std::string> all_lef_paths;
  all_lef_paths.reserve(assets.lef_paths.size() + 1);
  all_lef_paths.push_back(assets.tech_lef_path.string());
  for (const auto& lef_path : assets.lef_paths) {
    if (lef_path != assets.tech_lef_path) {
      all_lef_paths.push_back(lef_path.string());
    }
  }
  dm_config.set_lef_paths(all_lef_paths);
  dm_config.set_def_path(assets.def_path.string());

  std::vector<std::string> lib_paths;
  lib_paths.reserve(assets.lib_paths.size());
  for (const auto& lib_path : assets.lib_paths) {
    lib_paths.push_back(lib_path.string());
  }
  dm_config.set_lib_paths(lib_paths);
  dm_config.set_sdc_path(assets.sdc_path.string());

  const auto work_dir = io::ResolveClusteringOutputDir() / "real_tech_workspace";
  std::error_code error_code;
  std::filesystem::create_directories(work_dir, error_code);
  if (error_code) {
    error = "cannot create work dir: " + work_dir.string();
    return false;
  }
  icts_test::runtime::CurrentRuntime().config.set_work_dir(work_dir.string());
  dm_config.set_output_path((io::ResolveClusteringOutputDir() / "real_tech_output").string());

  auto* idb_builder = dmInst->get_idb_builder();
  if (idb_builder == nullptr) {
    error = "idb builder is null after LEF/DEF load";
    return false;
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
    error = "readClockData failed for SDC-declared clocks";
    return false;
  }
  return true;
}

}  // namespace

auto BuildRealTechSetupState() -> RealTechSetupState
{
  RealTechSetupState state;
  state.output_dir = io::ResolveClusteringOutputDir();
  std::error_code error_code;
  std::filesystem::create_directories(state.output_dir, error_code);

  std::vector<std::string> probe_errors;
  const auto candidates = CollectCandidateAssets();
  for (const auto& assets : candidates) {
    const auto& workspace_path = assets.workspace_path;

    std::string validation_error;
    if (!ValidateAssets(assets, validation_error)) {
      probe_errors.emplace_back(workspace_path.string() + " invalid: " + validation_error);
      continue;
    }

    state.assets_available = true;
    state.flow_script_path = assets.flow_script_path;
    state.cts_config_path = assets.cts_config_path;

    std::string load_error;
    if (LoadRealTechAssets(assets, load_error)) {
      state.mode = RealTechMode::kRealTech;
      state.setup_succeeded = true;
      state.source_label = "real_tech:" + workspace_path.string();
      state.summary = "loaded real tech/design from workspace " + workspace_path.string() + ", def=" + assets.def_path.filename().string()
                      + ", verilog=" + (assets.verilog_path.empty() ? std::string("<none>") : assets.verilog_path.filename().string());
      LOG_INFO << "RealTechSetup: " << state.summary;
      return state;
    }

    probe_errors.emplace_back(workspace_path.string() + " load failed: " + load_error);
  }

  std::ostringstream summary;
  summary << "use synthetic stand-in sweeps";
  if (!probe_errors.empty()) {
    summary << " (";
    for (std::size_t index = 0; index < probe_errors.size(); ++index) {
      if (index != 0) {
        summary << "; ";
      }
      summary << probe_errors.at(index);
    }
    summary << ")";
  } else {
    summary << " (no real-tech workspace candidates were found)";
  }
  state.mode = RealTechMode::kSyntheticLoads;
  state.source_label = "synthetic_standin";
  state.summary = summary.str();
  LOG_WARNING << "RealTechSetup: " << state.summary;
  return state;
}

}  // namespace icts_test::common::realtech::asset
