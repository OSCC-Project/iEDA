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
 * @file CharBuilderImpl.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Private Pimpl aggregate holding the CharBuilder characterization
 *        sweep state plus the 9 cooperating components (setup configurator,
 *        build orchestrator, pattern enumerator, topology planner, feasibility
 *        checker, pattern storage, circuit builder, STA sampler). Lives entirely
 *        behind the CharBuilder facade. All private nested types and shared
 *        data members of the algorithm live here at namespace
 *        `icts::char_builder::detail`.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ClockRouteSegmentRc.hh"

namespace icts {
class BufferingPattern;
class FastSTA;
class SchemaWriter;
class SegmentChar;
class Wrapper;
struct CharacterizationBufferCell;
}  // namespace icts

namespace icts::char_builder::detail {

class CharSetupConfigurator;
class CharBuildOrchestrator;
class CharPatternEnumerator;
class CharTopologyPlanner;
class CharFeasibilityChecker;
class CharPatternStorage;
class CharCircuitBuilder;
class CharStaSampler;

inline constexpr double kCapFeasibilityEpsilonPf = 1e-6;

struct BuildProgress
{
  double wirelength_um = 0.0;
  std::size_t estimated_patterns = 0;
  std::size_t estimated_sta_samples = 0;
  std::size_t evaluated_patterns = 0;
  std::size_t feasible_patterns = 0;
  std::size_t skipped_patterns_infeasible = 0;
  std::size_t skipped_load_points = 0;
  std::size_t skipped_sta_samples = 0;
  std::size_t executed_sta_samples = 0;
  std::size_t output_slew_overflow_samples = 0;
  std::size_t driven_cap_overflow_samples = 0;
  std::size_t driven_cap_overflow_load_points = 0;
  double max_observed_output_slew_ns = 0.0;
  unsigned max_observed_output_slew_idx = 0;
  double max_observed_driven_cap_pf = 0.0;
  unsigned max_observed_driven_cap_idx = 0;
};

struct TopologyBits
{
  std::uint64_t value = 0U;
};

struct TopologyDesc
{
  std::vector<double> wire_segments_um;
  std::vector<std::size_t> buffer_positions;
  bool has_terminal_branch_buffer = false;
};

struct StoredSampleIndices
{
  unsigned input_slew_idx = 0U;
  unsigned output_slew_idx = 0U;
  unsigned driven_cap_idx = 0U;
  unsigned load_cap_idx = 0U;
};

struct PatternFeasibility
{
  bool is_pattern_feasible = false;
  double max_load_pf = 0.0;
};

/**
 * @brief Pimpl aggregate for CharBuilder.
 *
 * Holds:
 *   - the full characterization sweep state (formerly private fields of CharBuilder);
 *   - the 8 cooperating components (one `std::unique_ptr` each);
 *   - sweep-grid metadata shared across multiple components.
 *
 * Components are friends so they can read/write the algorithm state directly,
 * and call cross-component helpers via the public accessors below.
 */
class CharBuilderImpl
{
  friend class CharSetupConfigurator;
  friend class CharBuildOrchestrator;
  friend class CharPatternEnumerator;
  friend class CharTopologyPlanner;
  friend class CharFeasibilityChecker;
  friend class CharPatternStorage;
  friend class CharCircuitBuilder;
  friend class CharStaSampler;

 public:
  CharBuilderImpl();
  ~CharBuilderImpl();
  CharBuilderImpl(const CharBuilderImpl&) = delete;
  CharBuilderImpl(CharBuilderImpl&&) = delete;
  auto operator=(const CharBuilderImpl&) -> CharBuilderImpl& = delete;
  auto operator=(CharBuilderImpl&&) -> CharBuilderImpl& = delete;

  auto setupConfigurator() -> CharSetupConfigurator&;
  auto buildOrchestrator() -> CharBuildOrchestrator&;
  auto patternEnumerator() -> CharPatternEnumerator&;
  auto topologyPlanner() -> CharTopologyPlanner&;
  auto feasibilityChecker() -> CharFeasibilityChecker&;
  auto patternStorage() -> CharPatternStorage&;
  auto circuitBuilder() -> CharCircuitBuilder&;
  auto staSampler() -> CharStaSampler&;

  auto segmentChars() const -> const std::vector<::icts::SegmentChar>& { return _segment_chars; }
  auto bufferingPatterns() const -> const std::vector<::icts::BufferingPattern>& { return _buffering_patterns; }
  auto wirelengthsUm() const -> const std::vector<double>& { return _wirelengths_um; }
  auto wirelengthIndices() const -> const std::vector<unsigned>& { return _wirelength_indices; }
  auto wirelengthUnitUm() const -> double { return _length_unit_um; }
  auto wirelengthUnitSource() const -> const std::string& { return _wirelength_unit_source; }
  auto wirelengthUnitDetail() const -> const std::string& { return _wirelength_unit_detail; }
  auto wirelengthIterations() const -> unsigned { return _wirelength_iterations; }
  auto maxSlew() const -> double { return _max_slew; }
  auto maxCap() const -> double { return _max_cap; }
  auto slewSteps() const -> unsigned { return _slew_steps; }
  auto capSteps() const -> unsigned { return _cap_steps; }
  auto routingLayer() const -> int { return _routing_layer; }
  auto wireWidth() const -> std::optional<double> { return _wire_width_um; }
  auto clockRouteSegmentRc() const -> const ::icts::ClockRouteSegmentRc& { return _clock_route_segment_rc; }
  auto characterizationBufferCells() const -> const std::vector<::icts::CharacterizationBufferCell>& { return _sorted_buffers; }
  auto executedStaSamples() const -> std::size_t { return _executed_sta_samples; }
  auto skippedStaSamples() const -> std::size_t { return _skipped_sta_samples; }
  auto outputSlewOverflowSamples() const -> std::size_t { return _output_slew_overflow_samples; }
  auto drivenCapOverflowSamples() const -> std::size_t { return _driven_cap_overflow_samples; }
  auto drivenCapOverflowLoadPoints() const -> std::size_t { return _driven_cap_overflow_load_points; }
  auto maxObservedOutputSlewNs() const -> double { return _max_observed_output_slew_ns; }
  auto maxObservedOutputSlewIdx() const -> unsigned { return _max_observed_output_slew_idx; }
  auto maxObservedDrivenCapPf() const -> double { return _max_observed_driven_cap_pf; }
  auto maxObservedDrivenCapIdx() const -> unsigned { return _max_observed_driven_cap_idx; }

 private:
  std::vector<::icts::CharacterizationBufferCell> _sorted_buffers;

  // Physical sweep grids kept in user units before discretization.
  std::vector<unsigned> _wirelength_indices;
  std::vector<double> _wirelengths_um;
  std::vector<double> _slews_to_test;
  std::vector<double> _loads_to_test;
  int _routing_layer = 0;
  std::optional<double> _wire_width_um = std::nullopt;
  ::icts::ClockRouteSegmentRc _clock_route_segment_rc;
  std::optional<std::int32_t> _dbu_per_um = std::nullopt;
  double _root_input_slew_ns = 0.0;
  double _max_slew = 0.0;
  double _max_cap = 0.0;
  double _max_length = 0.0;
  double _length_unit_um = 0.0;
  std::string _wirelength_unit_source;
  std::string _wirelength_unit_detail;
  unsigned _slew_steps = 15;
  unsigned _cap_steps = 15;
  unsigned _wirelength_iterations = 3;

  std::string _source_inst_name;
  std::string _source_in_pin;
  std::string _sink_inst_name;
  std::string _source_out_pin;
  std::string _sink_in_pin;
  std::string _timing_observation_pin;
  double _sink_input_cap_pf = 0.0;
  std::vector<std::string> _temp_inst_names;
  std::vector<std::string> _temp_net_names;
  std::string _char_clock_name;
  unsigned _char_circuit_id = 0;
  std::size_t _fast_sta_char_context_id = std::numeric_limits<std::size_t>::max();
  ::icts::Wrapper* _wrapper = nullptr;
  ::icts::FastSTA* _fast_sta = nullptr;
  ::icts::SchemaWriter* _reporter = nullptr;

  std::vector<::icts::SegmentChar> _segment_chars;
  std::vector<::icts::BufferingPattern> _buffering_patterns;
  unsigned _next_pattern_id = 0;
  std::size_t _executed_sta_samples = 0;
  std::size_t _skipped_sta_samples = 0;
  std::size_t _output_slew_overflow_samples = 0;
  std::size_t _driven_cap_overflow_samples = 0;
  std::size_t _driven_cap_overflow_load_points = 0;
  double _max_observed_output_slew_ns = 0.0;
  unsigned _max_observed_output_slew_idx = 0;
  double _max_observed_driven_cap_pf = 0.0;
  unsigned _max_observed_driven_cap_idx = 0;

  std::unique_ptr<CharSetupConfigurator> _setup_configurator;
  std::unique_ptr<CharBuildOrchestrator> _build_orchestrator;
  std::unique_ptr<CharPatternEnumerator> _pattern_enumerator;
  std::unique_ptr<CharTopologyPlanner> _topology_planner;
  std::unique_ptr<CharFeasibilityChecker> _feasibility_checker;
  std::unique_ptr<CharPatternStorage> _pattern_storage;
  std::unique_ptr<CharCircuitBuilder> _circuit_builder;
  std::unique_ptr<CharStaSampler> _sta_sampler;
};

}  // namespace icts::char_builder::detail
