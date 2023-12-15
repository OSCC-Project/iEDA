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
 * @file PyTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include <algorithm>
#include <vector>

#include "CtsPoint.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
#include "model/ModelFactory.hh"
#include "model/mplHelper/MplHelper.hh"

using ieda::Log;

namespace {

class PyTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "PyTest";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(PyTest, PyFitModel)
{
  LOG_INFO << "build PyTest for fit python model";
  auto* model_factory = new icts::ModelFactory();

  std::vector<double> x1 = {1, 2, 3, 4, 5};
  std::vector<double> x2 = {1, 2, 3, 4, 5};
  std::vector<std::vector<double>> x = {x1, x2};
  std::vector<double> y = {1, 2, 3, 4, 5};

  auto* linear_model = model_factory->pyFit(x, y, icts::FitType::kLinear);
  EXPECT_TRUE(linear_model->predict({1, 1}));

  auto* cat_model = model_factory->pyFit(x, y, icts::FitType::kCatBoost);
  EXPECT_TRUE(cat_model->predict({1, 1}));

  auto* xgb_model = model_factory->pyFit(x, y, icts::FitType::kXgBoost);
  EXPECT_TRUE(xgb_model->predict({1, 1}));

  delete model_factory;
}

TEST_F(PyTest, PyLoadModel)
{
  LOG_INFO << "build PyTest for load python model";
  auto* model_factory = new icts::ModelFactory();

  // auto* xgb_model = model_factory->pyLoad("./scripts/design/sky130_gcd/result/cts/model/chiplink_rx_clk.joblib.dat");
  // EXPECT_TRUE(xgb_model->predict({1, 0, 0, 0, 0, 11, 0.005172, 6, 0.0552891, 0.0561129}));

  delete model_factory;
}

TEST_F(PyTest, pyPlot)
{
  using Point = icts::Point;
  LOG_INFO << "build PyTest for python matplotlib";
  auto* mpl = new icts::MplHelper();

  Point point(1, 1);
  std::pair<Point, Point> segment = std::make_pair(Point(0, 2), Point(2, 0));
  std::vector<Point> poly = {Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2), Point(0, 0)};

  mpl->plot(point, "point");
  mpl->plot(segment.first, segment.second, "segment");
  mpl->plot(poly, "poly");
  mpl->saveFig("./poly.png");

  delete mpl;
}

}  // namespace
