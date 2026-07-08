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
 * @file CharacterizationRealTechExactRegression.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Shared helpers for exact real-tech characterization regression tests.
 */
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "characterization/Characterization.hh"
#include "database/characterization/SegmentChar.hh"
#include "database/characterization/ValueLattice.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"

namespace icts_test {

namespace realtech_fixture = characterization::realtech;

auto MakeExactRegressionCharBuilderContract() -> realtech_fixture::RuntimeCharBuilderContract;
auto MakeIterOneExperimentCharBuilderContract() -> realtech_fixture::RuntimeCharBuilderContract;

enum class FitBasisKind
{
  kLinear,
  kQuadratic,
};

auto FitBasisName(FitBasisKind basis_kind) -> std::string;
auto FitBasisSize(FitBasisKind basis_kind) -> std::size_t;
auto MakeFitBasis(FitBasisKind basis_kind, double s, double c) -> std::array<double, 6>;
auto TryFitSurfaceCoefficients(const std::vector<const icts::SegmentChar*>& group,
                               const std::function<double(const icts::SegmentChar&)>& metric_fn, const realtech_fixture::CharGrid& grid,
                               FitBasisKind basis_kind) -> std::optional<std::array<double, 6>>;
auto AppendIterOneFitReport(std::ostringstream& report_stream, const std::vector<icts::SegmentChar>& entries,
                            const realtech_fixture::CharGrid& grid, unsigned length_idx) -> void;

struct SegmentCompareKey
{
  std::string pattern_signature;
  unsigned input_slew_idx = 0U;
  unsigned output_slew_idx = 0U;
  unsigned driven_cap_idx = 0U;
  unsigned load_cap_idx = 0U;

  auto operator==(const SegmentCompareKey& rhs) const -> bool = default;
};

struct SegmentCompareKeyHash
{
  auto operator()(const SegmentCompareKey& key) const -> std::size_t
  {
    std::size_t seed = std::hash<std::string>{}(key.pattern_signature);
    seed ^= std::hash<unsigned>{}(key.input_slew_idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<unsigned>{}(key.output_slew_idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<unsigned>{}(key.driven_cap_idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<unsigned>{}(key.load_cap_idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct ComposeGapStats
{
  unsigned target_length_idx = 0U;
  std::size_t direct_count = 0U;
  std::size_t composed_count = 0U;
  std::size_t direct_duplicate_key_count = 0U;
  std::size_t matched_count = 0U;
  std::size_t missing_composed_count = 0U;
  std::size_t composed_lower_delay_count = 0U;
  std::size_t composed_higher_delay_count = 0U;
  std::size_t composed_lower_power_count = 0U;
  std::size_t composed_higher_power_count = 0U;
  double sum_direct_delay_ns = 0.0;
  double sum_composed_delay_ns = 0.0;
  double sum_abs_delay_delta_ns = 0.0;
  double sum_sq_delay_delta_ns = 0.0;
  double max_abs_delay_delta_ns = 0.0;
  double max_rel_delay_delta = 0.0;
  double sum_direct_power_w = 0.0;
  double sum_composed_power_w = 0.0;
  double sum_abs_power_delta_w = 0.0;
  double sum_sq_power_delta_w = 0.0;
  double max_abs_power_delta_w = 0.0;
  double max_rel_power_delta = 0.0;
  std::string worst_delay_example;
  std::string worst_power_example;
  std::string missing_composed_example;
};

auto MakeSegmentCompareKey(const realtech_fixture::SegmentFrontierContext& segment_context, const icts::SegmentChar& entry)
    -> SegmentCompareKey;
auto SafeRatio(double numerator, double denominator) -> double;
auto CompareComposedFrontierToDirect(unsigned target_length_idx, const std::vector<icts::SegmentChar>& direct_entries,
                                     const std::vector<icts::SegmentChar>& composed_entries,
                                     const realtech_fixture::SegmentFrontierContext& segment_context,
                                     const realtech_fixture::CharGrid& grid) -> ComposeGapStats;
auto AppendComposeGapStats(std::ostringstream& report_stream, const ComposeGapStats& stats, const realtech_fixture::CharGrid& grid) -> void;

struct FunctionalSurfaceModel
{
  std::string unit_key;
  FitBasisKind basis_kind = FitBasisKind::kLinear;
  std::array<double, 6> output_slew_coefficients{};
  std::array<double, 6> driven_cap_coefficients{};
  std::array<double, 6> delay_coefficients{};
  std::array<double, 6> power_coefficients{};
  std::array<double, 6> source_boundary_power_coefficients{};
};

struct StructuralCapOperator
{
  std::string unit_key;
  double alpha = 0.0;
  double eta_pf = 0.0;
  std::size_t sample_count = 0U;
  double max_abs_residual_pf = 0.0;
};

struct FunctionalSurfaceSample
{
  double output_slew_ns = 0.0;
  double driven_cap_pf = 0.0;
  double delay_ns = 0.0;
  double power_w = 0.0;
  double source_boundary_power_w = 0.0;
};

struct FunctionalComposePrediction
{
  bool is_valid = false;
  bool is_out_of_domain = false;
  bool converged = false;
  unsigned iterations = 0U;
  double residual = 0.0;
  double output_slew_ns = 0.0;
  double driven_cap_pf = 0.0;
  double delay_ns = 0.0;
  double power_w = 0.0;
  double source_boundary_power_w = 0.0;
};

auto MakeUnitPatternKey(const std::vector<double>& buffer_positions, const std::vector<std::string>& cell_masters) -> std::string;
auto MakeUnitPatternKey(const icts::BufferingPattern& pattern) -> std::string;
auto BuildFunctionalSurfaceModels(const std::vector<icts::SegmentChar>& entries,
                                  const realtech_fixture::SegmentFrontierContext& segment_context, const realtech_fixture::CharGrid& grid,
                                  FitBasisKind basis_kind) -> std::unordered_map<std::string, FunctionalSurfaceModel>;
auto BuildPhysicalStructuralCapOperators(const realtech_fixture::SegmentFrontierContext& segment_context,
                                         const icts::CharBuilder::Config& config, const realtech_fixture::CharGrid& grid)
    -> std::unordered_map<std::string, StructuralCapOperator>;
auto DecomposeToUnitPatternKeys(const icts::BufferingPattern& pattern, unsigned target_length_idx)
    -> std::optional<std::vector<std::string>>;
auto PredictFunctionalCompose(const std::vector<const FunctionalSurfaceModel*>& unit_models, double input_slew_ns, double load_cap_pf,
                              double max_slew_ns, double max_cap_pf) -> FunctionalComposePrediction;
auto PredictStructuralCapFunctionalCompose(const std::vector<const FunctionalSurfaceModel*>& unit_models,
                                           const std::vector<const StructuralCapOperator*>& cap_operators, double input_slew_ns,
                                           double load_cap_pf, double max_slew_ns, double max_cap_pf) -> FunctionalComposePrediction;

struct MetricGapAccumulator
{
  std::size_t count = 0U;
  std::size_t predicted_lower_count = 0U;
  std::size_t predicted_higher_count = 0U;
  double sum_direct = 0.0;
  double sum_predicted = 0.0;
  double sum_abs_delta = 0.0;
  double sum_sq_delta = 0.0;
  double max_abs_delta = 0.0;
  double max_rel_delta = 0.0;
  std::string worst_example;
};

struct FunctionComposeGapStats
{
  std::string source_label;
  unsigned target_length_idx = 0U;
  FitBasisKind basis_kind = FitBasisKind::kLinear;
  std::size_t direct_count = 0U;
  std::size_t decomposable_count = 0U;
  std::size_t missing_model_count = 0U;
  std::size_t convergence_failure_count = 0U;
  std::size_t invalid_prediction_count = 0U;
  std::size_t out_of_domain_count = 0U;
  std::size_t evaluated_count = 0U;
  double max_fixed_point_residual = 0.0;
  unsigned max_fixed_point_iterations = 0U;
  MetricGapAccumulator output_slew;
  MetricGapAccumulator driven_cap;
  MetricGapAccumulator delay;
  MetricGapAccumulator power;
  MetricGapAccumulator source_boundary_power;
  std::string missing_model_example;
  std::string convergence_failure_example;
  std::string invalid_prediction_example;
};

auto AnalyzeFunctionComposeGap(const std::string& source_label, unsigned target_length_idx, FitBasisKind basis_kind,
                               const std::vector<icts::SegmentChar>& direct_entries,
                               const realtech_fixture::SegmentFrontierContext& segment_context,
                               const std::unordered_map<std::string, FunctionalSurfaceModel>& model_by_unit_key,
                               const realtech_fixture::CharGrid& grid, double max_slew_ns, double max_cap_pf) -> FunctionComposeGapStats;
auto AnalyzeStructuralCapFunctionComposeGap(const std::string& source_label, unsigned target_length_idx, FitBasisKind basis_kind,
                                            const std::vector<icts::SegmentChar>& direct_entries,
                                            const realtech_fixture::SegmentFrontierContext& segment_context,
                                            const std::unordered_map<std::string, FunctionalSurfaceModel>& model_by_unit_key,
                                            const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key,
                                            const realtech_fixture::CharGrid& grid, const icts::UniformValueLattice& cap_lattice,
                                            double max_slew_ns, double max_cap_pf) -> FunctionComposeGapStats;
auto AppendFunctionComposeGapStats(std::ostringstream& report_stream, const FunctionComposeGapStats& stats,
                                   const realtech_fixture::CharGrid& grid) -> void;
auto AppendStructuralCapOperatorStats(std::ostringstream& report_stream,
                                      const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key) -> void;
auto AppendStructuralCapOperatorSampleGap(std::ostringstream& report_stream,
                                          const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key,
                                          const std::vector<icts::SegmentChar>& entries,
                                          const realtech_fixture::SegmentFrontierContext& segment_context,
                                          const realtech_fixture::CharGrid& grid, const icts::UniformValueLattice& cap_lattice) -> void;

}  // namespace icts_test
