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
 * @file TreeBuilderTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include <vector>

#include "TreeBuilderAux.hh"
#include "gtest/gtest.h"

namespace {
class TreeBuilderTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "TreeBuilderTest";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(TreeBuilderTest, GeomTest)
{
  using icts::bst::GeomCalc;
  using icts::bst::Pt;
  std::vector<Pt> poly = {Pt(14.419999999999995, 117.1),
                          Pt(12.222348146191434, 117.10000000000001),
                          Pt(12.105146073119494, 117.10000000000001),
                          Pt(10.36, 117.1),
                          Pt(10.36, 116.61827541858675),
                          Pt(10.36, 116.45292687949919),
                          Pt(10.36, 116.2),
                          Pt(12.052527055946209, 116.20000000000002),
                          Pt(12.383212964229324, 116.2),
                          Pt(14.42, 116.2),
                          Pt(14.419999999999998, 116.61827541858675),
                          Pt(14.419999999999995, 117.1)};
  GeomCalc::convexHull(poly);
  EXPECT_EQ(poly.size(), 4);
  poly = {Pt(111.86000000000001, 91.9),
          Pt(111.86, 94.14238973823346),
          Pt(111.86, 97.3),
          Pt(110.84921068871424, 97.3),
          Pt(110.32, 97.3),
          Pt(110.31999999999998, 93.9634039969406),
          Pt(110.31999999999998, 91.9),
          Pt(110.84921068871424, 91.9),
          Pt(111.86000000000001, 91.9)};
  GeomCalc::convexHull(poly);
  EXPECT_EQ(poly.size(), 4);

  using icts::BalanceClustering;
  using icts::Point;
  std::vector<Point> poly_t = {Point(462000, 649000), Point(701000, 649000), Point(771000, 579000), Point(771000, 504000),
                               Point(658000, 391000), Point(477000, 391000), Point(372000, 496000), Point(372000, 559000)};
  BalanceClustering::convexHull(poly_t);
  EXPECT_EQ(poly_t.size(), 8);
}

TEST_F(TreeBuilderTest, ParetoFrontTest)
{
  using icts::BalanceClustering;
  using Pt = icts::CtsPoint<double>;
  // random generate points with y = 1 / x
  std::vector<Pt> points;
  for (size_t i = 0; i < 20; ++i) {
    auto x = 1.0 * (i + 1) / 100;
    auto y = 1 / x;
    for (size_t j = 1; j < 10; ++j) {
      points.emplace_back(x + (std::rand() % (j * 10)) / 1000.0, y);
    }
  }
  auto pareto_front = BalanceClustering::paretoFront(points);
  // write to python file
  std::ofstream ofs("./pareto_front.py");
  ofs << "import matplotlib.pyplot as plt\n";
  ofs << "x = [";
  std::ranges::for_each(points, [&](const auto& point) { ofs << point.x() << ","; });
  ofs << "]\n";
  ofs << "y = [";
  std::ranges::for_each(points, [&](const auto& point) { ofs << point.y() << ","; });
  ofs << "]\n";
  ofs << "plt.scatter(x, y)\n";
  ofs << "x = [";
  std::ranges::for_each(pareto_front, [&](const auto& point) { ofs << point.x() << ","; });
  ofs << "]\n";
  ofs << "y = [";
  std::ranges::for_each(pareto_front, [&](const auto& point) { ofs << point.y() << ","; });
  ofs << "]\n";
  ofs << "plt.scatter(x, y, c='r', marker='o')\n";
  ofs << "plt.savefig('pareto_front.png')\n";
  ofs.close();
}

TEST_F(TreeBuilderTest, LocalLegalizationTest)
{
  auto* load1 = TreeBuilder::genBufInst("load1", Point(2606905, 3009850));
  std::vector<Pin*> load_pins = {load1->get_load_pin()};
  TreeBuilder::localPlace(load_pins);
  LOG_INFO << "load1 location: " << load1->get_location();
}

TEST_F(TreeBuilderTest, FixedTreeBuilderTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  double skew_bound = 0.08;
  tree_builder.runFixedTest(skew_bound);
}

TEST_F(TreeBuilderTest, RegressionTreeBuilderTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  std::vector<double> skew_bound_list = {0.08, 0.01, 0.005};
  size_t case_num = 500;
  // design DB unit is 2000
  EnvInfo env_info{0, 150000, 0, 150000, 10, 40, 0, 0, 0, 0};
  std::ranges::for_each(skew_bound_list, [&](const double& skew_bound) {
    auto data_set = tree_builder.runRegressTest(env_info, case_num, skew_bound);

    auto suffix = "skew_" + std::to_string(skew_bound);
    // drop "0" in the suffix end
    while (suffix.back() == '0') {
      suffix.pop_back();
    }

    auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/file/" + suffix;
    auto method_list = {TreeBuilder::funcName(TreeBuilder::fluteTree), TreeBuilder::funcName(TreeBuilder::shallowLightTree),
                        TreeBuilder::funcName(TreeBuilder::boundSkewTree), TreeBuilder::funcName(TreeBuilder::bstSaltTree),
                        TreeBuilder::funcName(TreeBuilder::cbsTree)};
    auto topo_type_list = {
        TopoTypeToString(TopoType::kGreedyDist),
        TopoTypeToString(TopoType::kGreedyMerge),
        TopoTypeToString(TopoType::kBiCluster),
        TopoTypeToString(TopoType::kBiPartition),
    };
    // all data
    data_set.writeCSV(method_list, topo_type_list, dir, "regression_" + suffix + ".csv");

    // relative compare
    std::ranges::for_each(method_list, [&](const auto& target_method) { data_set.writeReduceCSV(target_method, dir, suffix); });
  });
}

TEST_F(TreeBuilderTest, LowBoundEstimationTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  std::vector<double> skew_bound_list = {0.08, 0.01, 0.005};
  size_t case_num = 1000;
  // design DB unit is 2000
  std::ranges::for_each(skew_bound_list, [&](const double& skew_bound) {
    EnvInfo env_info{0, 150000, 0, 150000, 10, 40, 0.005, 0.01, skew_bound / 100, skew_bound / 10};
    auto suffix = "skew_" + std::to_string(skew_bound);
    // drop "0" in the suffix end
    while (suffix.back() == '0') {
      suffix.pop_back();
    }

    auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/file/" + suffix;

    tree_builder.runEstimationTest(env_info, case_num, skew_bound, dir, suffix);
  });
}

}  // namespace