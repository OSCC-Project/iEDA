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
 * @file ModelFactory.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>
#include <vector>

namespace icts {

enum class FitType
{
  kLinear,
  kCatBoost,
  kXgBoost
};

class ModelFactory
{
 public:
  std::vector<double> solvePolynomialRealRoots(const std::vector<double>& coeffs) const;

  std::vector<double> cppLinearModel(const std::vector<std::vector<double>>& x, const std::vector<double>& y) const;

  double criticalBufWireLen(const double& alpha, const double& beta, const double& gamma, const double& r, const double& c,
                            const double& cap_pin);
  std::pair<std::pair<double, double>, std::pair<double, double>> criticalSteinerWireLen(const double& alpha, const double& beta,
                                                                                         const double& gamma, const double& r,
                                                                                         const double& c, const double& cap_pin,
                                                                                         const double& input_slew, const double& cap_load);
  double criticalError(const double& r, const double& c, const double& x, const double& cap_load, const double& cap_pin_low,
                       const double& cap_pin_high, const double& input_slew, const double& gamma, const double& beta_i,
                       const double& beta_k);
};
}  // namespace icts