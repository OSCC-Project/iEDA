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
 * @file CharFeasibilityChecker.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder feasibility-checking component. Provides the
 *        characterization-buffer-cell lookup, the per-segment wire
 *        capacitance estimate, and the per-topology static-load feasibility
 *        analysis used to prune patterns whose driver cap budget cannot
 *        accommodate the wire + downstream load.
 */

#pragma once

#include <string>
#include <vector>

namespace icts {
struct CharacterizationBufferCell;
}  // namespace icts

namespace icts::char_builder::detail {

class CharBuilderImpl;
struct TopologyDesc;
struct PatternFeasibility;

class CharFeasibilityChecker
{
 public:
  explicit CharFeasibilityChecker(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharFeasibilityChecker() = default;
  CharFeasibilityChecker(const CharFeasibilityChecker&) = delete;
  auto operator=(const CharFeasibilityChecker&) -> CharFeasibilityChecker& = delete;

  auto findCharacterizationBufferCell(const std::string& cell_master) const -> const ::icts::CharacterizationBufferCell*;
  auto calcClockRouteWireCapPf(double wirelength_um) const -> double;
  auto analyzePatternFeasibility(const TopologyDesc& topo, const std::vector<std::string>& buf_masters) const -> PatternFeasibility;

 private:
  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
