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
 * @file CharBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Segment characterization builder facade. The sweep state and the
 *        cooperating components (setup configurator, build orchestrator,
 *        pattern enumerator, topology planner, feasibility checker, pattern
 *        storage, circuit builder, STA sampler) live behind a Pimpl boundary
 *        at `char_builder::detail::CharBuilderImpl`.
 */

#pragma once
// IWYU pragma: private, include "characterization/builder/CharBuilder.hh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ValueLattice.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "routing/ClockRouteSegmentRc.hh"

namespace icts::char_builder::detail {
class CharBuilderImpl;
}  // namespace icts::char_builder::detail

namespace icts {

class SchemaWriter;
class BufferingPattern;
class FastSTA;
class SegmentChar;
class Wrapper;
struct CharBuilderInput
{
  std::vector<std::string> buffer_types;
  std::vector<CharacterizationBufferCell> characterization_buffer_cells;
  ClockRouteSegmentRc clock_route_segment_rc;
  std::optional<std::int32_t> dbu_per_um = std::nullopt;
  double root_input_slew_ns = 0.0;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
};

struct CharBuilderConfig
{
  std::optional<double> wirelength_unit_um = std::nullopt;
  std::optional<unsigned> wirelength_iterations = std::nullopt;
  std::optional<std::vector<unsigned>> wirelength_indices = std::nullopt;
  bool allow_auto_wirelength_unit = false;
  std::optional<double> max_slew_ns = std::nullopt;
  std::optional<double> max_cap_pf = std::nullopt;
  std::optional<double> char_buf_redundancy_pct = std::nullopt;
  std::optional<unsigned> slew_steps = std::nullopt;
  std::optional<unsigned> cap_steps = std::nullopt;
  std::optional<int> routing_layer = std::nullopt;
  std::optional<double> wire_width_um = std::nullopt;
};

/**
 * @brief Builds segment characterization entries by enumerating buffering
 *        patterns and querying STA/PA through CTSAPI.
 *
 * Physical values (slew ns, cap pF, length um) are discretized to integer
 * bin indices in [1, *_steps] for storage in CharCore/SegmentChar.
 */
class CharBuilder
{
 public:
  using Input = CharBuilderInput;
  using Config = CharBuilderConfig;

  CharBuilder();
  ~CharBuilder();
  CharBuilder(const CharBuilder&) = delete;
  CharBuilder(CharBuilder&&) noexcept;
  auto operator=(const CharBuilder&) -> CharBuilder& = delete;
  auto operator=(CharBuilder&&) noexcept -> CharBuilder&;

  auto init(const Input& input, const Config& config) -> void;
  auto build() -> void;

  auto get_segment_chars() const -> const std::vector<SegmentChar>&;
  auto get_buffering_patterns() const -> const std::vector<BufferingPattern>&;
  auto get_wirelengths_um() const -> const std::vector<double>&;
  auto get_wirelength_indices() const -> const std::vector<unsigned>&;
  auto get_wirelength_unit_um() const -> double;
  auto get_wirelength_unit_source() const -> const std::string&;
  auto get_wirelength_unit_detail() const -> const std::string&;
  auto get_wirelength_iterations() const -> unsigned;
  auto get_max_slew() const -> double;
  auto get_max_cap() const -> double;
  auto get_slew_steps() const -> unsigned;
  auto get_cap_steps() const -> unsigned;
  auto get_routing_layer() const -> int;
  auto get_wire_width() const -> std::optional<double>;
  auto get_clock_route_segment_rc() const -> const ClockRouteSegmentRc&;
  auto get_characterization_buffer_cells() const -> const std::vector<CharacterizationBufferCell>&;
  auto get_length_lattice() const -> UniformValueLattice;
  auto get_slew_lattice() const -> UniformValueLattice;
  auto get_cap_lattice() const -> UniformValueLattice;
  auto get_executed_sta_samples() const -> std::size_t;
  auto get_skipped_sta_samples() const -> std::size_t;
  auto get_output_slew_overflow_samples() const -> std::size_t;
  auto get_driven_cap_overflow_samples() const -> std::size_t;
  auto get_driven_cap_overflow_load_points() const -> std::size_t;
  auto get_max_observed_output_slew_ns() const -> double;
  auto get_max_observed_output_slew_idx() const -> unsigned;
  auto get_max_observed_driven_cap_pf() const -> double;
  auto get_max_observed_driven_cap_idx() const -> unsigned;

 private:
  std::unique_ptr<char_builder::detail::CharBuilderImpl> _impl;
};
}  // namespace icts
