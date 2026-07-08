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
 * @file CharBuilderImpl.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder Pimpl implementation: constructor wires up the 8
 *        cooperating components; destructor needs to see complete component
 *        types so the unique_ptr destructors compile.
 */

#include "characterization/builder/CharBuilderImpl.hh"

#include <memory>

#include "BufferingPattern.hh"
#include "SegmentChar.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuildOrchestrator.hh"
#include "characterization/builder/CharFeasibilityChecker.hh"
#include "characterization/builder/CharSetupConfigurator.hh"
#include "characterization/builder/CharTopologyPlanner.hh"
#include "characterization/circuit/CharCircuitBuilder.hh"
#include "characterization/pattern/CharPatternEnumerator.hh"
#include "characterization/pattern/CharPatternStorage.hh"
#include "characterization/sampling/CharStaSampler.hh"

namespace icts::char_builder::detail {

CharBuilderImpl::CharBuilderImpl()
    : _setup_configurator(std::make_unique<CharSetupConfigurator>(*this)),
      _build_orchestrator(std::make_unique<CharBuildOrchestrator>(*this)),
      _pattern_enumerator(std::make_unique<CharPatternEnumerator>(*this)),
      _topology_planner(std::make_unique<CharTopologyPlanner>(*this)),
      _feasibility_checker(std::make_unique<CharFeasibilityChecker>(*this)),
      _pattern_storage(std::make_unique<CharPatternStorage>(*this)),
      _circuit_builder(std::make_unique<CharCircuitBuilder>(*this)),
      _sta_sampler(std::make_unique<CharStaSampler>(*this))
{
}

CharBuilderImpl::~CharBuilderImpl() = default;

auto CharBuilderImpl::setupConfigurator() -> CharSetupConfigurator&
{
  return *_setup_configurator;
}

auto CharBuilderImpl::buildOrchestrator() -> CharBuildOrchestrator&
{
  return *_build_orchestrator;
}

auto CharBuilderImpl::patternEnumerator() -> CharPatternEnumerator&
{
  return *_pattern_enumerator;
}

auto CharBuilderImpl::topologyPlanner() -> CharTopologyPlanner&
{
  return *_topology_planner;
}

auto CharBuilderImpl::feasibilityChecker() -> CharFeasibilityChecker&
{
  return *_feasibility_checker;
}

auto CharBuilderImpl::patternStorage() -> CharPatternStorage&
{
  return *_pattern_storage;
}

auto CharBuilderImpl::circuitBuilder() -> CharCircuitBuilder&
{
  return *_circuit_builder;
}

auto CharBuilderImpl::staSampler() -> CharStaSampler&
{
  return *_sta_sampler;
}

}  // namespace icts::char_builder::detail
