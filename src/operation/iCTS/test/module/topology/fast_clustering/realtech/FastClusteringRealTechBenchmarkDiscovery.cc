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
 * @file FastClusteringRealTechBenchmarkDiscovery.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Case and technology asset discovery for the fast-clustering real-tech benchmark.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "FastClusteringRealTechBenchmarkFixture.hh"

namespace icts_test::fast_clustering::realtech {
namespace {

auto EndsWith(std::string_view text, std::string_view suffix) -> bool
{
  return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

auto TrimAscii(const std::string& text) -> std::string
{
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1U);
}

auto ToLowerAscii(std::string text) -> std::string
{
  std::ranges::transform(text, text.begin(), [](unsigned char ch) -> char { return static_cast<char>(std::tolower(ch)); });
  return text;
}

auto TryParseVerilogTopModule(const std::filesystem::path& verilog_path) -> std::string
{
  std::ifstream input_stream(verilog_path);
  if (!input_stream) {
    return {};
  }

  std::string line;
  while (std::getline(input_stream, line)) {
    auto trimmed = TrimAscii(line);
    if (!trimmed.starts_with("module ")) {
      continue;
    }
    std::istringstream line_stream(trimmed);
    std::string keyword;
    std::string module_name;
    line_stream >> keyword >> module_name;
    while (!module_name.empty() && (module_name.back() == '(' || module_name.back() == ';')) {
      module_name.pop_back();
    }
    return module_name;
  }
  return {};
}

auto StripDefSuffix(const std::string& filename) -> std::string
{
  if (EndsWith(filename, ".def.gz")) {
    return filename.substr(0U, filename.size() - std::string_view(".def.gz").size());
  }
  if (EndsWith(filename, ".def")) {
    return filename.substr(0U, filename.size() - std::string_view(".def").size());
  }
  return filename;
}

auto ReadEnvPath(std::string_view key) -> std::filesystem::path
{
  const auto env_key = std::string(key);
  const char* value = std::getenv(env_key.c_str());
  return value == nullptr || *value == '\0' ? std::filesystem::path{} : std::filesystem::path(value);
}

auto FirstExistingPath(const std::vector<std::filesystem::path>& candidates) -> std::filesystem::path
{
  std::error_code error_code;
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate, error_code)) {
      return candidate;
    }
  }
  return {};
}

}  // namespace

auto ContainsClockToken(std::string_view name) -> bool
{
  const auto lowered = ToLowerAscii(std::string(name));
  return lowered == "clk" || lowered == "clock" || lowered.find("clk") != std::string::npos || lowered.find("clock") != std::string::npos;
}

auto DiscoverBenchmarkCases() -> std::vector<BenchmarkCase>
{
  std::vector<BenchmarkCase> cases;
  const std::filesystem::path root(kBenchmarkRoot);
  std::error_code error_code;
  if (!std::filesystem::exists(root, error_code)) {
    return cases;
  }

  std::vector<std::filesystem::path> case_dirs;
  for (std::filesystem::directory_iterator it(root, error_code), end; !error_code && it != end; it.increment(error_code)) {
    if (it->is_directory(error_code)) {
      case_dirs.push_back(it->path());
    }
  }
  std::ranges::sort(case_dirs);

  for (const auto& case_dir : case_dirs) {
    const auto output_dir = case_dir / "place_dreamplace" / "output";
    if (!std::filesystem::is_directory(output_dir, error_code)) {
      continue;
    }

    std::vector<std::filesystem::path> def_paths;
    for (std::filesystem::directory_iterator it(output_dir, error_code), end; !error_code && it != end; it.increment(error_code)) {
      if (!it->is_regular_file(error_code)) {
        continue;
      }
      const auto filename = it->path().filename().string();
      if ((EndsWith(filename, "_place.def.gz") || EndsWith(filename, "_place.def"))) {
        def_paths.push_back(it->path());
      }
    }
    std::ranges::sort(def_paths);

    for (const auto& def_path : def_paths) {
      const auto def_base = StripDefSuffix(def_path.filename().string());
      const auto verilog_path = output_dir / (def_base + ".v");
      if (!std::filesystem::exists(verilog_path, error_code)) {
        continue;
      }
      const auto top_module = TryParseVerilogTopModule(verilog_path);
      if (top_module.empty()) {
        continue;
      }
      cases.push_back(BenchmarkCase{.index = cases.size() + 1U,
                                    .case_name = case_dir.filename().string(),
                                    .top_module = top_module,
                                    .case_dir = case_dir,
                                    .def_path = def_path,
                                    .verilog_path = verilog_path});
      break;
    }
    if (cases.size() == kRequiredCaseCount) {
      break;
    }
  }
  return cases;
}

auto ResolveTechAssets() -> TechAssets
{
  TechAssets assets;
  assets.cts_config_path = std::filesystem::path(kCtsConfigPath);
  assets.sdc_path = std::filesystem::path(kDefaultSdcPath);
  assets.pdk_root = FirstExistingPath({
      ReadEnvPath("ICTS_REALTECH_PDK_DIR"),
      ReadEnvPath("PDK_DIR"),
      std::filesystem::path("/home/liweiguo/pdk/icsprout55-pdk"),
      std::filesystem::path("/home/liweiguo/pdk"),
  });

  assets.tech_lef_path = FirstExistingPath({
      assets.pdk_root / "prtech/techLEF/N551P6M_ieda.lef",
      assets.pdk_root / "prtech/techLEF/N551P6M_ecos.lef",
  });
  assets.lef_paths = {
      FirstExistingPath({
          assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR_ieda.lef",
          assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR_ecos.lef",
      }),
      FirstExistingPath({
          assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL_ieda.lef",
          assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL_ecos.lef",
      }),
  };
  assets.lib_paths = {
      assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib",
      assets.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_ss_rcworst_1p08_125_nldm.lib",
  };
  return assets;
}

auto ValidateTechAssets(const TechAssets& assets, std::string& error) -> bool
{
  std::vector<std::string> missing;
  auto check_file = [&missing](std::string_view label, const std::filesystem::path& path) -> void {
    std::error_code error_code;
    if (path.empty() || !std::filesystem::exists(path, error_code)) {
      missing.push_back(std::string(label) + "=" + path.string());
    }
  };

  check_file("pdk_root", assets.pdk_root);
  check_file("cts_config", assets.cts_config_path);
  check_file("sdc", assets.sdc_path);
  check_file("tech_lef", assets.tech_lef_path);
  for (const auto& lef_path : assets.lef_paths) {
    check_file("lef", lef_path);
  }
  for (const auto& lib_path : assets.lib_paths) {
    check_file("lib", lib_path);
  }
  if (missing.empty()) {
    return true;
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < missing.size(); ++index) {
    if (index > 0U) {
      stream << "; ";
    }
    stream << missing.at(index);
  }
  error = stream.str();
  return false;
}

}  // namespace icts_test::fast_clustering::realtech
