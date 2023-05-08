#include <vector>

#include "gtest/gtest.h"
#include "log/Log.hh"
#include "model/ModelFactory.h"
#include "model/mplHelper/MplHelper.h"

using ieda::Log;

namespace {

class PyTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(PyTest, PyFitModel) {
  LOG_INFO << "build PyTest for fit python model";
  auto* model_factory = new icts::ModelFactory();

  std::vector<double> x1 = {1, 2, 3, 4, 5};
  std::vector<double> x2 = {1, 2, 3, 4, 5};
  std::vector<std::vector<double>> x = {x1, x2};
  std::vector<double> y = {1, 2, 3, 4, 5};

  auto* linear_model = model_factory->pyFit(x, y, icts::FitType::kLINEAR);
  EXPECT_TRUE(linear_model->predict({1, 1}));

  auto* cat_model = model_factory->pyFit(x, y, icts::FitType::kCATBOOST);
  EXPECT_TRUE(cat_model->predict({1, 1}));

  auto* xgb_model = model_factory->pyFit(x, y, icts::FitType::kXGBOOST);
  EXPECT_TRUE(xgb_model->predict({1, 1}));

  delete model_factory;
}

TEST_F(PyTest, PyLoadModel) {
  LOG_INFO << "build PyTest for load python model";
  auto* model_factory = new icts::ModelFactory();

  auto* xgb_model = model_factory->pyLoad(
      "/home/liweiguo/project/irefactor/scripts/28nm/t28/result/cts/model/"
      "chiplink_rx_clk.joblib.dat");
  EXPECT_TRUE(xgb_model->predict(
      {1, 0, 0, 0, 0, 11, 0.005172, 6, 0.0552891, 0.0561129}));

  delete model_factory;
}

TEST_F(PyTest, pyPlot) {
  LOG_INFO << "build PyTest for python matplotlib";
  auto* mpl = new icts::MplHelper();

  icts::Point point(1, 1);
  icts::Segment segment({0, 2}, {2, 0});
  icts::Polygon poly({{0, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 0}});

  mpl->plot(point, "point");
  mpl->plot(segment, "segment");
  mpl->plot(poly, "poly");
  mpl->saveFig("./poly.png");

  delete mpl;
}

}  // namespace