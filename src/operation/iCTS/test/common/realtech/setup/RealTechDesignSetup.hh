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
 * @file RealTechDesignSetup.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared real-tech setup helpers used by iCTS real-tech tests.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/dataset/TestDataset.hh"

namespace icts {
class Pin;
}

namespace icts_test::common::realtech {

enum class RealTechMode : std::uint8_t
{
  kRealTech,
  kSyntheticLoads
};

struct RealTechSetupState
{
  RealTechMode mode = RealTechMode::kSyntheticLoads;
  bool assets_available = false;
  bool setup_succeeded = false;
  std::filesystem::path output_dir;
  std::filesystem::path flow_script_path;
  std::filesystem::path cts_config_path;
  std::string source_label;
  std::string summary;
};

struct RealPinCapProbe
{
  std::string net_name;
  std::string inst_name;
  std::string cell_master;
  std::string pin_name;
  bool is_clock_net = false;
  double pre_timing_cap_pf = 0.0;
};

struct RealClockNetSelection
{
  std::string clock_name;
  std::string net_name;
  icts::Pin* source = nullptr;
  std::vector<icts::Pin*> sinks;
  std::size_t source_net_load_count = 0U;
  bool is_def_clock_net = false;
  std::size_t clock_like_load_pin_count = 0U;
};

auto EnsureRealTechSetup() -> const RealTechSetupState&;
auto MakeRealTechOrSyntheticLoads(std::size_t target_count, unsigned seed, std::string& source_label) -> GeneratedPins;
auto TryFindRepresentativeRealPinCapProbe() -> std::optional<RealPinCapProbe>;
auto SelectLargestDefClock(std::size_t max_count, std::size_t min_required_load_count) -> std::optional<RealClockNetSelection>;

}  // namespace icts_test::common::realtech
