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
 * @file AnalyticalFitTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Unit tests for analytical characterization least-squares fitting.
 */

#include <gtest/gtest.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "analytical_characterization/AnalyticalFit.hh"
#include "analytical_characterization/AnalyticalModel.hh"

namespace icts_test {
namespace {

using icts::analytical::AnalyticalFitConfig;
using icts::analytical::AnalyticalFitSample;
using icts::analytical::AnalyticalMetric;
using icts::analytical::AnalyticalModelBasis;
using icts::analytical::FitAnalyticalSurface;

TEST(AnalyticalFitTest, FitsExactAffineSurface)
{
  std::vector<AnalyticalFitSample> samples;
  for (double slew : {0.02, 0.04, 0.06}) {
    for (double cap : {0.01, 0.02, 0.03}) {
      samples.push_back(AnalyticalFitSample{.input_slew_ns = slew, .load_cap_pf = cap, .value = 0.1 + 2.0 * slew + 3.0 * cap});
    }
  }

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kDelay;
  config.basis = AnalyticalModelBasis::kAffine;
  config.min_r2 = 0.999999;
  config.require_monotonic = true;
  const auto result = FitAnalyticalSurface(samples, config);

  ASSERT_TRUE(result.summary.success) << result.summary.failure_reason;
  if (!result.output.model.has_value()) {
    ADD_FAILURE() << "Expected fitted model.";
    return;
  }
  const auto& model = result.output.model.value();
  const auto predicted = model.evaluate(0.04, 0.02);
  if (!predicted.has_value()) {
    ADD_FAILURE() << "Expected in-domain prediction.";
    return;
  }
  EXPECT_NEAR(predicted.value(), 0.24, 1e-12);
  EXPECT_NEAR(model.quality.rmse, 0.0, 1e-12);
  EXPECT_TRUE(model.quality.monotonicity_passed);
}

TEST(AnalyticalFitTest, FitsExactQuadraticSurface)
{
  std::vector<AnalyticalFitSample> samples;
  for (double slew : {0.02, 0.04, 0.06, 0.08}) {
    for (double cap : {0.01, 0.02, 0.03, 0.04}) {
      samples.push_back(AnalyticalFitSample{
          .input_slew_ns = slew,
          .load_cap_pf = cap,
          .value = 0.05 + slew + 2.0 * cap + 0.5 * slew * cap + 3.0 * slew * slew + 4.0 * cap * cap,
      });
    }
  }

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kPower;
  config.basis = AnalyticalModelBasis::kQuadratic;
  config.min_r2 = 0.999999;
  const auto result = FitAnalyticalSurface(samples, config);

  ASSERT_TRUE(result.summary.success) << result.summary.failure_reason;
  if (!result.output.model.has_value()) {
    ADD_FAILURE() << "Expected fitted model.";
    return;
  }
  const auto& model = result.output.model.value();
  const auto predicted = model.evaluate(0.06, 0.03);
  if (!predicted.has_value()) {
    ADD_FAILURE() << "Expected in-domain prediction.";
    return;
  }
  EXPECT_NEAR(predicted.value(), 0.05 + 0.06 + 2.0 * 0.03 + 0.5 * 0.06 * 0.03 + 3.0 * 0.06 * 0.06 + 4.0 * 0.03 * 0.03, 1e-12);
}

TEST(AnalyticalFitTest, RejectsInsufficientSamples)
{
  const std::vector<AnalyticalFitSample> samples = {
      {.input_slew_ns = 0.02, .load_cap_pf = 0.01, .value = 0.1},
      {.input_slew_ns = 0.04, .load_cap_pf = 0.02, .value = 0.2},
  };

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kDelay;
  config.basis = AnalyticalModelBasis::kAffine;
  const auto result = FitAnalyticalSurface(samples, config);

  EXPECT_FALSE(result.summary.success);
  EXPECT_EQ(result.summary.failure_reason, "insufficient_samples");
}

TEST(AnalyticalFitTest, RejectsOutOfDomainEvaluation)
{
  std::vector<AnalyticalFitSample> samples;
  for (double slew : {0.02, 0.04, 0.06}) {
    for (double cap : {0.01, 0.02, 0.03}) {
      samples.push_back(AnalyticalFitSample{.input_slew_ns = slew, .load_cap_pf = cap, .value = 0.1 + slew + cap});
    }
  }

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kDelay;
  config.basis = AnalyticalModelBasis::kAffine;
  const auto result = FitAnalyticalSurface(samples, config);

  ASSERT_TRUE(result.summary.success);
  if (!result.output.model.has_value()) {
    ADD_FAILURE() << "Expected fitted model.";
    return;
  }
  const auto& model = result.output.model.value();
  EXPECT_FALSE(model.evaluate(0.10, 0.02).has_value());
}

TEST(AnalyticalFitTest, RejectsNonMonotoneSurfaceWhenRequired)
{
  std::vector<AnalyticalFitSample> samples;
  for (double slew : {0.02, 0.04, 0.06}) {
    for (double cap : {0.01, 0.02, 0.03}) {
      samples.push_back(AnalyticalFitSample{.input_slew_ns = slew, .load_cap_pf = cap, .value = 1.0 - slew - cap});
    }
  }

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kOutputSlew;
  config.basis = AnalyticalModelBasis::kAffine;
  config.require_monotonic = true;
  const auto result = FitAnalyticalSurface(samples, config);

  EXPECT_FALSE(result.summary.success);
  EXPECT_EQ(result.summary.failure_reason, "monotonicity_check_failed");
}

TEST(AnalyticalFitTest, RecordsBucketAwareResidual)
{
  std::vector<AnalyticalFitSample> samples;
  for (double slew : {0.02, 0.04, 0.06}) {
    for (double cap : {0.01, 0.02, 0.03}) {
      samples.push_back(AnalyticalFitSample{.input_slew_ns = slew, .load_cap_pf = cap, .value = 0.10});
    }
  }
  samples.back().value = 0.11;

  AnalyticalFitConfig config;
  config.metric = AnalyticalMetric::kOutputSlew;
  config.basis = AnalyticalModelBasis::kAffine;
  config.bucket_size = 0.01;
  const auto result = FitAnalyticalSurface(samples, config);

  ASSERT_TRUE(result.summary.success);
  if (!result.output.model.has_value()) {
    ADD_FAILURE() << "Expected fitted model.";
    return;
  }
  const auto& model = result.output.model.value();
  EXPECT_GT(model.quality.max_bucket_residual, 0.0);
}

}  // namespace
}  // namespace icts_test
