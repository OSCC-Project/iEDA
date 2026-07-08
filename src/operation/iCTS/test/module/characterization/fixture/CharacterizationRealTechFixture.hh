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
 * @file CharacterizationRealTechFixture.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Shared helpers for real-tech characterization tests.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "characterization/Characterization.hh"
#include "database/characterization/BufferingPattern.hh"
#include "database/characterization/HTreeTopologyChar.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/SegmentChar.hh"

namespace icts_test::common::logging {
class ScopedLogFile;
}  // namespace icts_test::common::logging

namespace icts_test::characterization::realtech {

inline constexpr std::size_t kReportExampleLimit = 3U;
inline constexpr double kLeafLevelLengthUm = 50.0;
inline constexpr double kMidLevelLengthUm = 100.0;
inline constexpr double kRootLevelLengthUm = 200.0;
inline constexpr double kExactLeafLevelLengthUm = 25.0;
inline constexpr double kExactMidLevelLengthUm = 50.0;
inline constexpr double kExactRootLevelLengthUm = 100.0;
inline constexpr double kRealTechCharWirelengthUnitUm = 25.0;
inline constexpr unsigned kRealTechCharWirelengthIterations = 3U;
inline constexpr unsigned kRealTechCharSlewSteps = 15U;
inline constexpr unsigned kRealTechCharCapSteps = 15U;

struct ConfigState
{
  double skew_bound = 0.0;
  double max_buf_tran = 0.0;
  double root_input_slew = 0.0;
  double max_sink_tran = 0.0;
  double max_cap = 0.0;
  bool has_max_buf_tran = false;
  bool has_max_cap = false;
  double max_length = 0.0;
  double wirelength_unit_um = 0.0;
  unsigned wirelength_iterations = 0U;
  unsigned slew_steps = 0U;
  unsigned cap_steps = 0U;
  double wire_width = 0.0;
  unsigned max_fanout = 0U;
  std::vector<unsigned> routing_layers;
  std::vector<std::string> buffer_types;
  double char_buf_redundancy_pct = 0.0;
  bool force_branch_buffer = false;
  unsigned htree_depth_explore_window = 4U;
  bool enable_sink_clustering = true;
  std::string work_dir;
  std::string log_file;
  std::string visualization_dir;
  std::string statistics_dir;
};

struct RuntimeCharBuilderContract
{
  icts::CharBuilder::Input input;
  icts::CharBuilder::Config config;
};

auto CaptureConfigState() -> ConfigState;
auto ApplyConfigState(const ConfigState& state) -> void;
auto MakeRuntimeCharBuilderContract() -> RuntimeCharBuilderContract;
auto MakeRealTechCharConfigState(const ConfigState& baseline_state, std::optional<std::vector<std::string>> buffer_types,
                                 double max_buf_tran_ns, double max_cap_pf, bool omit_wirelength_unit, bool force_branch_buffer = false)
    -> ConfigState;

struct RealTechCharFixture
{
  RealTechCharFixture();
  ~RealTechCharFixture();

  RealTechCharFixture(const RealTechCharFixture&) = delete;
  auto operator=(const RealTechCharFixture&) -> RealTechCharFixture& = delete;
  RealTechCharFixture(RealTechCharFixture&&) = delete;
  auto operator=(RealTechCharFixture&&) -> RealTechCharFixture& = delete;

  auto prepare(const std::string& scenario_name, std::optional<std::vector<std::string>> buffer_types, double max_buf_tran_ns,
               double max_cap_pf, bool omit_wirelength_unit = false, bool force_branch_buffer = false) -> std::optional<std::string>;

 private:
  auto restore() -> void;

  bool _is_prepared = false;
  std::optional<ConfigState> _original_config_state;
  std::unique_ptr<::icts_test::common::logging::ScopedLogFile> _cts_log_guard;
};

struct BufferLimitInfo
{
  std::string cell_master;
  std::string input_pin;
  std::string output_pin;
  double port_slew_limit_ns = 0.0;
  double table_slew_limit_ns = 0.0;
  double port_cap_limit_pf = 0.0;
  double table_cap_limit_pf = 0.0;
};

struct CharGrid
{
  double length_step_um = 0.0;
  double slew_step_ns = 0.0;
  double cap_step_pf = 0.0;
};

struct SegmentCharLatticeSummary
{
  std::size_t total_entries = 0U;
  std::size_t out_of_range_entries = 0U;
  std::size_t length_overflow_entries = 0U;
  std::size_t input_slew_overflow_entries = 0U;
  std::size_t output_slew_overflow_entries = 0U;
  std::size_t driven_cap_overflow_entries = 0U;
  std::size_t load_cap_overflow_entries = 0U;
  unsigned max_length_idx = 0U;
  unsigned max_input_slew_idx = 0U;
  unsigned max_output_slew_idx = 0U;
  unsigned max_driven_cap_idx = 0U;
  unsigned max_load_cap_idx = 0U;
};

struct HTreeStageSummary
{
  std::string label;
  std::vector<icts::HTreeTopologyChar> raw_entries;
  std::vector<icts::HTreeTopologyChar> frontier_entries;
};

struct SegmentFrontierContext
{
  std::unordered_map<icts::PatternId, icts::BufferingPattern> patterns;
  std::unordered_map<icts::PatternId, icts::PatternCompositionState> composition_states;
  unsigned next_pattern_id = 0U;
};

struct HTreeFrontierContext
{
  std::unordered_map<icts::PatternId, icts::PatternCompositionState> composition_states;
  unsigned next_pattern_id = 0U;
};

auto JoinStrings(const std::vector<std::string>& values) -> std::string;
auto WriteScenarioLog(const std::string& scenario_name, const std::string& file_name, const std::string& content) -> bool;
auto CollectConfiguredBufferLimitInfo() -> std::vector<BufferLimitInfo>;
auto CollectUsableBufferMasters(const std::vector<BufferLimitInfo>& infos) -> std::vector<std::string>;
auto LookupBufferInfo(const std::vector<BufferLimitInfo>& infos, const std::string& cell_master) -> const BufferLimitInfo*;
auto MinPositiveResolvedLimit(const std::vector<BufferLimitInfo>& infos, const std::vector<std::string>& selected_masters, bool for_slew)
    -> double;
auto ResolveDefaultWirelengthUnitUm(const std::vector<BufferLimitInfo>& infos, const std::vector<std::string>& selected_masters) -> double;

template <class PredicateT>
inline auto CollectMastersByPredicate(const std::vector<BufferLimitInfo>& infos, const PredicateT& predicate) -> std::vector<std::string>
{
  std::vector<std::string> masters;
  for (const auto& info : infos) {
    if (predicate(info)) {
      masters.push_back(info.cell_master);
    }
  }
  return masters;
}

template <class CharT>
inline auto SortCharsForReport(std::vector<CharT>& chars) -> void
{
  std::ranges::sort(chars, [](const CharT& lhs, const CharT& rhs) -> bool {
    if (lhs.get_input_slew_idx() != rhs.get_input_slew_idx()) {
      return lhs.get_input_slew_idx() < rhs.get_input_slew_idx();
    }
    if (lhs.get_driven_cap_idx() != rhs.get_driven_cap_idx()) {
      return lhs.get_driven_cap_idx() < rhs.get_driven_cap_idx();
    }
    if (lhs.get_output_slew_idx() != rhs.get_output_slew_idx()) {
      return lhs.get_output_slew_idx() < rhs.get_output_slew_idx();
    }
    if (lhs.get_load_cap_idx() != rhs.get_load_cap_idx()) {
      return lhs.get_load_cap_idx() > rhs.get_load_cap_idx();
    }
    if (lhs.get_delay() != rhs.get_delay()) {
      return lhs.get_delay() < rhs.get_delay();
    }
    if (lhs.get_power() != rhs.get_power()) {
      return lhs.get_power() < rhs.get_power();
    }
    return lhs.get_pattern_id().pack() < rhs.get_pattern_id().pack();
  });
}

auto BuildSegmentFrontierContext(const std::vector<icts::BufferingPattern>& patterns) -> SegmentFrontierContext;
auto BuildSegmentStateFrontier(const std::vector<icts::SegmentChar>& chars, const SegmentFrontierContext& context)
    -> std::vector<icts::SegmentChar>;
auto BuildHTreeStateFrontier(const std::vector<icts::HTreeTopologyChar>& chars, const HTreeFrontierContext& context)
    -> std::vector<icts::HTreeTopologyChar>;

auto MakeLengthIndex(double length_um, double length_step_um) -> unsigned;
auto CalcCharGrid(const icts::CharBuilder& builder) -> CharGrid;
auto SummarizeSegmentCharLattice(const std::vector<icts::SegmentChar>& chars, const icts::CharBuilder& builder)
    -> SegmentCharLatticeSummary;
auto FormatSegmentCharLatticeSummary(const SegmentCharLatticeSummary& summary, const icts::CharBuilder& builder) -> std::string;
auto ComposeSegmentEntriesExact(const std::vector<icts::SegmentChar>& upstream, const std::vector<icts::SegmentChar>& downstream,
                                SegmentFrontierContext& context) -> std::vector<icts::SegmentChar>;
auto BuildSegmentLengthFrontiers(const std::vector<icts::SegmentChar>& chars, const SegmentFrontierContext& context)
    -> std::unordered_map<unsigned, std::vector<icts::SegmentChar>>;
auto SynthesizeSegmentFrontierExactOnly(std::unordered_map<unsigned, std::vector<icts::SegmentChar>>& frontier_by_length,
                                        unsigned target_length_idx, SegmentFrontierContext& context) -> bool;
auto MakeHTreeSeedEntries(const std::vector<icts::SegmentChar>& segment_frontier, const SegmentFrontierContext& segment_context,
                          HTreeFrontierContext& htree_context) -> std::vector<icts::HTreeTopologyChar>;
auto ComposeHTreeEntriesExact(const std::vector<icts::HTreeTopologyChar>& upstream, const std::vector<icts::HTreeTopologyChar>& downstream,
                              HTreeFrontierContext& htree_context) -> std::vector<icts::HTreeTopologyChar>;
auto CountPositivePower(const std::vector<icts::SegmentChar>& chars) -> std::size_t;
auto FormatSegmentChar(const icts::SegmentChar& entry, const CharGrid& grid) -> std::string;
auto FormatHTreeChar(const icts::HTreeTopologyChar& entry, const CharGrid& grid) -> std::string;

template <class CharT, class FormatFn>
inline auto AppendExamples(std::ostringstream& output_stream, const std::string& prefix, std::vector<CharT> entries,
                           const FormatFn& format_fn) -> void
{
  SortCharsForReport(entries);
  for (std::size_t index = 0; index < std::min(kReportExampleLimit, entries.size()); ++index) {
    output_stream << prefix << format_fn(entries.at(index)) << "\n";
  }
}

auto SelectBestHTreeChar(const std::vector<icts::HTreeTopologyChar>& entries) -> std::optional<icts::HTreeTopologyChar>;

}  // namespace icts_test::characterization::realtech
