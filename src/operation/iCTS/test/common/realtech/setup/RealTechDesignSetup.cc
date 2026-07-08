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
 * @file RealTechDesignSetup.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Facade for shared real-tech setup helpers used by linear-clustering tests.
 */

#include "common/realtech/setup/RealTechDesignSetup.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <compare>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Flow.hh"
#include "IdbDesign.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "Log.hh"
#include "common/CTSTestRuntime.hh"
#include "common/dataset/TestDataset.hh"
#include "common/realtech/asset/RealTechAssetLoader.hh"
#include "common/realtech/load/RealTechLoadFactory.hh"
#include "database/design/Clock.hh"
#include "database/design/Design.hh"
#include "database/io/Wrapper.hh"
#include "idm.h"

namespace icts_test::common::realtech {
namespace {

struct DefClockCandidate
{
  std::string net_name;
  std::size_t load_count = 0U;
  bool is_def_clock_net = false;
  std::size_t clock_like_load_pin_count = 0U;
};

auto ToLowerAscii(std::string text) -> std::string
{
  std::ranges::transform(text, text.begin(), [](unsigned char character) -> char { return static_cast<char>(std::tolower(character)); });
  return text;
}

auto ContainsClockToken(std::string_view name) -> bool
{
  const auto lowered_name = ToLowerAscii(std::string(name));
  return lowered_name.find("clk") != std::string::npos || lowered_name.find("clock") != std::string::npos;
}

auto IsClockLikePinName(std::string_view name) -> bool
{
  const auto lowered_name = ToLowerAscii(std::string(name));
  return lowered_name == "ck" || lowered_name == "clk" || lowered_name == "clock" || lowered_name == "cp";
}

auto CountClockLikeLoadPins(idb::IdbNet* net) -> std::size_t
{
  if (net == nullptr) {
    return 0U;
  }

  return static_cast<std::size_t>(std::ranges::count_if(
      net->get_load_pins(), [](idb::IdbPin* pin) -> bool { return pin != nullptr && IsClockLikePinName(pin->get_pin_name()); }));
}

auto CalcCandidatePriority(const DefClockCandidate& candidate) -> unsigned
{
  if (candidate.is_def_clock_net) {
    return 4U;
  }
  if (candidate.clock_like_load_pin_count > 0U) {
    return 3U;
  }
  if (ContainsClockToken(candidate.net_name)) {
    return 2U;
  }
  return 1U;
}

auto CollectDefClockCandidates(std::size_t min_required_load_count) -> std::vector<DefClockCandidate>
{
  std::vector<DefClockCandidate> candidates;
  auto* idb_design = dmInst->get_idb_design();
  auto* net_list = idb_design != nullptr ? idb_design->get_net_list() : nullptr;
  if (net_list == nullptr) {
    return candidates;
  }

  for (auto* net : net_list->get_net_list()) {
    if (net == nullptr) {
      continue;
    }
    const auto load_count = net->get_load_pins().size();
    if (load_count < min_required_load_count) {
      continue;
    }

    candidates.push_back(DefClockCandidate{
        .net_name = net->get_net_name(),
        .load_count = load_count,
        .is_def_clock_net = net->is_clock(),
        .clock_like_load_pin_count = CountClockLikeLoadPins(net),
    });
  }

  std::ranges::sort(candidates, [](const DefClockCandidate& lhs, const DefClockCandidate& rhs) -> bool {
    const unsigned lhs_priority = CalcCandidatePriority(lhs);
    const unsigned rhs_priority = CalcCandidatePriority(rhs);
    if (lhs_priority != rhs_priority) {
      return lhs_priority > rhs_priority;
    }
    if (lhs.load_count != rhs.load_count) {
      return lhs.load_count > rhs.load_count;
    }
    if (lhs.clock_like_load_pin_count != rhs.clock_like_load_pin_count) {
      return lhs.clock_like_load_pin_count > rhs.clock_like_load_pin_count;
    }
    return lhs.net_name < rhs.net_name;
  });
  return candidates;
}

auto SampleLoadsForRealTechClock(const std::vector<icts::Pin*>& loads, std::size_t max_count) -> std::vector<icts::Pin*>
{
  if (loads.size() <= max_count) {
    return loads;
  }

  std::vector<icts::Pin*> sampled_loads;
  sampled_loads.reserve(max_count);
  for (std::size_t sample_index = 0; sample_index < max_count; ++sample_index) {
    const std::size_t source_index = sample_index * loads.size() / max_count;
    sampled_loads.push_back(loads.at(source_index));
  }
  return sampled_loads;
}

auto TryMaterializeClockCandidate(const DefClockCandidate& candidate, std::size_t max_count, std::size_t min_required_load_count)
    -> std::optional<RealClockNetSelection>
{
  icts_test::runtime::CurrentRuntime().design.reset();
  auto* requested_clock = icts_test::runtime::CurrentRuntime().design.makeClock("def_clock:" + candidate.net_name, candidate.net_name);
  if (requested_clock != nullptr) {
    requested_clock->set_clock_name("def_clock:" + candidate.net_name);
    requested_clock->set_clock_net_name(candidate.net_name);
  }
  icts_test::runtime::CurrentRuntime().wrapper.read(icts_test::runtime::CurrentRuntime().design,
                                                    icts_test::runtime::CurrentRuntime().reporter);

  const auto clocks = icts_test::runtime::CurrentRuntime().design.get_clocks();
  if (clocks.empty() || clocks.front() == nullptr) {
    return std::nullopt;
  }

  auto* selected_clock = clocks.front();
  if (selected_clock->get_clock_source() == nullptr || selected_clock->get_loads().size() < min_required_load_count) {
    return std::nullopt;
  }

  auto sampled_sinks = SampleLoadsForRealTechClock(selected_clock->get_loads(), max_count);
  if (sampled_sinks.size() < min_required_load_count) {
    return std::nullopt;
  }

  return RealClockNetSelection{
      .clock_name = selected_clock->get_clock_name(),
      .net_name = selected_clock->get_clock_net_name(),
      .source = selected_clock->get_clock_source(),
      .sinks = std::move(sampled_sinks),
      .source_net_load_count = selected_clock->get_loads().size(),
      .is_def_clock_net = candidate.is_def_clock_net,
      .clock_like_load_pin_count = candidate.clock_like_load_pin_count,
  };
}

}  // namespace

auto EnsureRealTechSetup() -> const RealTechSetupState&
{
  static const RealTechSetupState setup_state = asset::BuildRealTechSetupState();
  return setup_state;
}

auto TryFindRepresentativeRealPinCapProbe() -> std::optional<RealPinCapProbe>
{
  return asset::TryFindRepresentativeRealPinCapProbe();
}

auto MakeRealTechOrSyntheticLoads(std::size_t target_count, unsigned seed, std::string& source_label) -> GeneratedPins
{
  const auto& setup_state = EnsureRealTechSetup();
  if (setup_state.mode == RealTechMode::kRealTech && setup_state.setup_succeeded) {
    auto real_loads = load::MakeRealDesignLoads(target_count, source_label, seed);
    if (!real_loads.loads.empty()) {
      return real_loads;
    }
    LOG_WARNING << "RealTechSetup: real design load extraction failed, use synthetic stand-in.";
  }

  return load::MakeSyntheticLoads(target_count, source_label, seed);
}

auto SelectLargestDefClock(std::size_t max_count, std::size_t min_required_load_count) -> std::optional<RealClockNetSelection>
{
  if (min_required_load_count == 0U) {
    return std::nullopt;
  }

  const auto candidates = CollectDefClockCandidates(min_required_load_count);
  for (const auto& candidate : candidates) {
    if (auto selection = TryMaterializeClockCandidate(candidate, max_count, min_required_load_count); selection.has_value()) {
      return selection;
    }
  }
  return std::nullopt;
}

}  // namespace icts_test::common::realtech
