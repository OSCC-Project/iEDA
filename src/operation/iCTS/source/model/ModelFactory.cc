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
 * @file ModelFactory.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "ModelFactory.hh"

#include <Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/Polynomials>

#ifdef PY_MODEL
#include "PyModel.h"
#endif
namespace icts {

std::vector<double> ModelFactory::solvePolynomialRealRoots(const std::vector<double>& coeffs) const
{
  int poly_order = coeffs.size() - 1;
  Eigen::VectorXd poly_coeffs(poly_order + 1);
  for (int i = 0; i < poly_order + 1; i++) {
    poly_coeffs(i) = coeffs[poly_order - i];
  }
  Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
  solver.compute(poly_coeffs);
  std::vector<double> result;
  solver.realRoots(result);
  return result;
}

std::vector<double> ModelFactory::cppLinearModel(const std::vector<std::vector<double>>& x, const std::vector<double>& y) const
{
  int m = x.size();
  int n = x[0].size();

  Eigen::MatrixXd t_x(n, m + 1);
  Eigen::VectorXd t_y(n);

  for (int i = 0; i < n; i++) {
    t_x(i, 0) = 1;
    for (int j = 0; j < m; j++) {
      t_x(i, j + 1) = x[j][i];
    }
    t_y(i) = y[i];
  }

  Eigen::VectorXd coeffs = (t_x.transpose() * t_x).ldlt().solve(t_x.transpose() * t_y);

  std::vector<double> result(m + 1);
  for (int i = 0; i < m + 1; i++) {
    result[i] = coeffs(i);
  }

  return result;
}
#ifdef PY_MODEL
/**
 * @brief Python interface for timing model
 *
 * @param x (m x n)
 * @param y (n)
 */

double ModelBase::predict(const std::vector<double>& x) const
{
  return pyPredict(x, _model);
}

ModelBase* ModelFactory::pyFit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, const FitType& fit_type) const
{
  PyObject* model;
  switch (fit_type) {
    case FitType::kLinear:
      model = pyLinearModel(x, y);
      break;
    case FitType::kCatBoost:
      model = pyCatBoostModel(x, y);
      break;
    case FitType::kXgBoost:
      model = pyXGBoostModel(x, y);
      break;
    default:
      model = pyLinearModel(x, y);
      break;
  }
  return new ModelBase(model);
}

ModelBase* ModelFactory::pyLoad(const std::string& model_path) const
{
  auto* model = pyLoadModel(model_path);
  return new ModelBase(model);
}
#endif
}  // namespace icts