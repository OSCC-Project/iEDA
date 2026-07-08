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
 * @file AnalyticalCharacterization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Facade for building analytical characterization catalogs from segment samples.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "AnalyticalModel.hh"
#include "routing/ClockRouteSegmentRc.hh"

namespace icts {
class BufferingPattern;
class CharBuilder;
class SegmentChar;
class UniformValueLattice;
struct PatternId;
}  // namespace icts

namespace icts::analytical {

struct AnalyticalCharacterizationConfig
{
  AnalyticalModelBasis output_slew_basis = AnalyticalModelBasis::kAffine;
  AnalyticalModelBasis delay_basis = AnalyticalModelBasis::kAffine;
  AnalyticalModelBasis power_basis = AnalyticalModelBasis::kAffine;
  AnalyticalModelBasis source_boundary_power_basis = AnalyticalModelBasis::kAffine;
  bool prefer_exact_structural_cap = false;
  double length_unit_um = 0.0;
  ClockRouteSegmentRc clock_route_segment_rc;
  std::unordered_map<std::string, double> buffer_input_cap_pf_by_cell_master;
  double max_output_slew_abs_residual_ns = 0.0;
  double max_output_slew_bucket_residual = 0.0;
  double max_delay_relative_residual = 0.0;
  double max_power_relative_residual = 0.0;
  double min_r2 = 0.0;
  bool require_monotonic_output_slew = true;
  bool require_monotonic_delay = true;
  bool require_monotonic_power = false;
  bool require_monotonic_source_boundary_power = false;
  bool allow_sparse_constant_model = false;
};

struct AnalyticalCharacterizationFailure
{
  AnalyticalModelKey key;
  AnalyticalMetric metric = AnalyticalMetric::kDelay;
  std::string reason;
};

struct AnalyticalCharacterizationOutput
{
  AnalyticalModelCatalog catalog;
};

struct AnalyticalCharacterizationSummary
{
  bool success = false;
  std::size_t model_set_count = 0U;
  std::size_t rejected_fit_count = 0U;
  std::size_t structural_cap_operator_count = 0U;
  std::vector<AnalyticalCharacterizationFailure> failures;
};

struct AnalyticalCharacterizationBuild
{
  AnalyticalCharacterizationOutput output;
  AnalyticalCharacterizationSummary summary;
};

class AnalyticalCharacterization
{
 public:
  static auto buildFromCharBuilder(const CharBuilder& char_builder, const AnalyticalCharacterizationConfig& config)
      -> AnalyticalCharacterizationBuild;
  static auto buildFromSegmentChars(const std::vector<SegmentChar>& segment_chars, const std::vector<BufferingPattern>& buffering_patterns,
                                    const UniformValueLattice& slew_lattice, const UniformValueLattice& cap_lattice,
                                    const AnalyticalCharacterizationConfig& config) -> AnalyticalCharacterizationBuild;
};

auto BuildBucketCompatibleStructuralCapOperator(PatternId pattern_id, const std::vector<BufferingPattern>& buffering_patterns,
                                                const std::vector<SegmentChar>& grouped_chars, const UniformValueLattice& cap_lattice)
    -> std::optional<StructuralCapOperator>;

}  // namespace icts::analytical
