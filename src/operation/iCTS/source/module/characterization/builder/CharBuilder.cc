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
 * @file CharBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Segment characterization builder facade. Forwards each public method
 *        to the Pimpl implementation at `char_builder::detail::CharBuilderImpl`,
 *        which dispatches to the cooperating components.
 */

#include "characterization/builder/CharBuilder.hh"

#include <memory>

#include "characterization/builder/CharBuildOrchestrator.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "characterization/builder/CharSetupConfigurator.hh"

namespace icts {
class BufferingPattern;
class SegmentChar;
struct ClockRouteSegmentRc;
}  // namespace icts

namespace icts {

CharBuilder::CharBuilder() : _impl(std::make_unique<char_builder::detail::CharBuilderImpl>())
{
}

CharBuilder::~CharBuilder() = default;

CharBuilder::CharBuilder(CharBuilder&&) noexcept = default;

auto CharBuilder::operator=(CharBuilder&&) noexcept -> CharBuilder& = default;

auto CharBuilder::init(const Input& input, const Config& config) -> void
{
  _impl->setupConfigurator().init(input, config);
}

auto CharBuilder::build() -> void
{
  _impl->buildOrchestrator().build();
}

auto CharBuilder::get_segment_chars() const -> const std::vector<SegmentChar>&
{
  return _impl->segmentChars();
}
auto CharBuilder::get_buffering_patterns() const -> const std::vector<BufferingPattern>&
{
  return _impl->bufferingPatterns();
}
auto CharBuilder::get_wirelengths_um() const -> const std::vector<double>&
{
  return _impl->wirelengthsUm();
}
auto CharBuilder::get_wirelength_indices() const -> const std::vector<unsigned>&
{
  return _impl->wirelengthIndices();
}
auto CharBuilder::get_wirelength_unit_um() const -> double
{
  return _impl->wirelengthUnitUm();
}
auto CharBuilder::get_wirelength_unit_source() const -> const std::string&
{
  return _impl->wirelengthUnitSource();
}
auto CharBuilder::get_wirelength_unit_detail() const -> const std::string&
{
  return _impl->wirelengthUnitDetail();
}
auto CharBuilder::get_wirelength_iterations() const -> unsigned
{
  return _impl->wirelengthIterations();
}
auto CharBuilder::get_max_slew() const -> double
{
  return _impl->maxSlew();
}
auto CharBuilder::get_max_cap() const -> double
{
  return _impl->maxCap();
}
auto CharBuilder::get_slew_steps() const -> unsigned
{
  return _impl->slewSteps();
}
auto CharBuilder::get_cap_steps() const -> unsigned
{
  return _impl->capSteps();
}
auto CharBuilder::get_routing_layer() const -> int
{
  return _impl->routingLayer();
}
auto CharBuilder::get_wire_width() const -> std::optional<double>
{
  return _impl->wireWidth();
}
auto CharBuilder::get_clock_route_segment_rc() const -> const ClockRouteSegmentRc&
{
  return _impl->clockRouteSegmentRc();
}
auto CharBuilder::get_characterization_buffer_cells() const -> const std::vector<CharacterizationBufferCell>&
{
  return _impl->characterizationBufferCells();
}
auto CharBuilder::get_length_lattice() const -> UniformValueLattice
{
  return UniformValueLattice(_impl->wirelengthUnitUm(), _impl->wirelengthIterations());
}
auto CharBuilder::get_slew_lattice() const -> UniformValueLattice
{
  return UniformValueLattice::buildFromMax(_impl->maxSlew(), _impl->slewSteps());
}
auto CharBuilder::get_cap_lattice() const -> UniformValueLattice
{
  return UniformValueLattice::buildFromMax(_impl->maxCap(), _impl->capSteps());
}
auto CharBuilder::get_executed_sta_samples() const -> std::size_t
{
  return _impl->executedStaSamples();
}
auto CharBuilder::get_skipped_sta_samples() const -> std::size_t
{
  return _impl->skippedStaSamples();
}
auto CharBuilder::get_output_slew_overflow_samples() const -> std::size_t
{
  return _impl->outputSlewOverflowSamples();
}
auto CharBuilder::get_driven_cap_overflow_samples() const -> std::size_t
{
  return _impl->drivenCapOverflowSamples();
}
auto CharBuilder::get_driven_cap_overflow_load_points() const -> std::size_t
{
  return _impl->drivenCapOverflowLoadPoints();
}
auto CharBuilder::get_max_observed_output_slew_ns() const -> double
{
  return _impl->maxObservedOutputSlewNs();
}
auto CharBuilder::get_max_observed_output_slew_idx() const -> unsigned
{
  return _impl->maxObservedOutputSlewIdx();
}
auto CharBuilder::get_max_observed_driven_cap_pf() const -> double
{
  return _impl->maxObservedDrivenCapPf();
}
auto CharBuilder::get_max_observed_driven_cap_idx() const -> unsigned
{
  return _impl->maxObservedDrivenCapIdx();
}

}  // namespace icts
