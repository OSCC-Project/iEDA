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
  auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file/";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream ofs(dir + "pareto_front.py");
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

TEST_F(TreeBuilderTest, PolynomialRealRootsTest)
{
  auto* model_factory = new icts::ModelFactory();
  auto coeffs = std::vector<double>{1, 2, 1};
  auto roots = model_factory->solvePolynomialRealRoots(coeffs);
  for (auto root : roots) {
    LOG_INFO << "Root: " << root;
  }
}

TEST_F(TreeBuilderTest, CriticalWireLenTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/design/eval/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/design/eval/iEDA_config/cts_default_config.json");
  auto libs = CTSAPIInst.getAllBufferLibs();
  auto* model_factory = new icts::ModelFactory();
  auto slew_in = 0.025;
  auto cap_load = 0.02;
  auto r = TimingPropagator::getUnitRes();
  auto c = TimingPropagator::getUnitCap();
  for (auto lib : libs) {
    LOG_INFO << "Lib: " << lib->get_cell_master();
    auto coeffs = lib->get_delay_coef();
    auto cap_pin = lib->get_init_cap();
    auto crit_buf_wl = model_factory->criticalBufWireLen(coeffs[2], coeffs[1], coeffs[0], r, c, cap_pin);
    auto crit_est_wl = model_factory->criticalBufWireLen(coeffs[2], coeffs[1], coeffs[0], r, c, cap_load);
    auto crit_pair = model_factory->criticalSteinerWireLen(coeffs[2], coeffs[1], coeffs[0], r, c, cap_pin, slew_in, cap_load);
    LOG_INFO << "Critical Buffer Wirelength: " << crit_buf_wl;
    LOG_INFO << "Critical EstSteiner Wirelength: " << crit_est_wl;
    if (crit_pair.first.second > 0) {
      LOG_INFO << "CWE_WL_1: " << crit_pair.first.first;
      LOG_INFO << "CWE_x_1: " << crit_pair.first.second;
    }
    if (crit_pair.second.second > 0) {
      LOG_INFO << "CWE_WL_2: " << crit_pair.second.first;
      LOG_INFO << "CWE_x_2: " << crit_pair.second.second;
    }
    LOG_INFO << "--------------------------------";
  }
  std::string output_csv = "./error.csv";
  std::ofstream ofs(output_csv);
  ofs << "x,i,k,j,size_i,size_k,size_j,error\n";
  for (double x = 0.1; x < 1.0; x += 0.1) {
    for (size_t j = 0; j < libs.size(); ++j) {
      for (size_t k = j; k < libs.size(); ++k) {
        for (size_t i = k; i < libs.size(); ++i) {
          auto lib_i = libs[i];
          auto lib_k = libs[k];
          auto lib_j = libs[j];
          auto coeffs_i = lib_i->get_delay_coef();
          auto coeffs_k = lib_k->get_delay_coef();
          auto cap_pin_j = lib_j->get_init_cap();
          auto cap_pin_k = lib_k->get_init_cap();
          auto error
              = model_factory->criticalError(r, c, x, cap_load, cap_pin_j, cap_pin_k, slew_in, coeffs_k[0], coeffs_i[1], coeffs_k[1]);
          ofs << x << "," << i << "," << k << "," << j << "," << lib_i->get_cell_master() << "," << lib_k->get_cell_master() << ","
              << lib_j->get_cell_master() << "," << error << "\n";
        }
      }
    }
  }
  ofs.close();
}

TEST_F(TreeBuilderTest, CellLinearDelayModelTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/design/eval/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/design/eval/iEDA_config/cts_default_config.json");
  auto libs = CTSAPIInst.getAllBufferLibs();
  auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file/";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  for (auto lib : libs) {
    LOG_INFO << "Lib: " << lib->get_cell_master();
    std::ofstream ofs(dir + lib->get_cell_master() + ".csv");
    ofs << "slew_in,cap_load,slew_out,delay\n";
    auto index_list = lib->get_index_list();
    auto slew_in_list = index_list.front();
    auto cap_load_list = index_list.back();
    auto slew_out_list = lib->get_slew_values();
    auto delay_list = lib->get_delay_values();
    for (size_t i = 0; i < slew_in_list.size(); ++i) {
      for (size_t j = 0; j < cap_load_list.size(); ++j) {
        ofs << slew_in_list[i] << "," << cap_load_list[j] << "," << slew_out_list[i * cap_load_list.size() + j] << ","
            << delay_list[i * cap_load_list.size() + j] << "\n";
      }
    }
    ofs.close();
  }
  LOG_INFO << "cell data write done";
}

TEST_F(TreeBuilderTest, SALTTest)
{
  std::vector<double> load_x
      = {193124, 193123, 193123, 193124, 193111, 193113, 193122, 193117, 193124, 193123, 193121, 193125, 204960, 193123, 213639,
         193130, 193129, 210840, 193135, 193134, 193132, 193131, 193123, 193123, 193128, 193126, 193123, 193126, 193123, 193124,
         174160, 193120, 193123, 193123, 193123, 193123, 193133, 193123, 213640, 193127, 213641, 193123, 213640, 192628, 192631};
  std::vector<double> load_y
      = {209000, 195703, 195702, 223400, 195698, 195698, 182000, 195698, 195692, 195694, 195698, 195698, 195698, 195689, 195698,
         195698, 195698, 195698, 195698, 195698, 195698, 195698, 195691, 195693, 195698, 195699, 195696, 195698, 195697, 195698,
         195698, 195698, 195705, 195704, 195701, 195700, 195698, 195690, 195697, 195698, 195698, 195698, 195698, 183849, 205500};

  auto* driver = TreeBuilder::genBufInst("driver", Point(195161, 196258));
  std::vector<Pin*> load_pins;
  for (size_t i = 0; i < load_x.size(); ++i) {
    auto* load = TreeBuilder::genBufInst("load_" + std::to_string(i), Point(load_x[i], load_y[i]));
    load_pins.push_back(load->get_load_pin());
  }
  TreeBuilder::shallowLightTree("net", driver->get_driver_pin(), load_pins);
}

TEST_F(TreeBuilderTest, LocalLegalizationTest)
{
  auto* load1 = TreeBuilder::genBufInst("load1", Point(2606905, 3009850));
  auto* load2 = TreeBuilder::genBufInst("load2", Point(2606905, 3009850));
  std::vector<Pin*> load_pins = {load1->get_load_pin(), load2->get_load_pin()};
  TreeBuilder::localPlace(load_pins);
  LOG_INFO << "load1 location: " << load1->get_location();
  LOG_INFO << "load2 location: " << load2->get_location();
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

    auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file/" + suffix;
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

    auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file/" + suffix;

    tree_builder.runEstimationTest(env_info, case_num, skew_bound, dir, suffix);
  });
}

TEST_F(TreeBuilderTest, IterativeFixSkewTest)
{
  TreeBuilderAux tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                              "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  // std::vector<double> skew_bound_list = {0.08, 0.01, 0.005};
  std::vector<double> skew_bound_list = {0.005};
  size_t case_num = 10000;
  // design DB unit is 2000
  std::ranges::for_each(skew_bound_list, [&](const double& skew_bound) {
    EnvInfo env_info{0, 150000, 0, 150000, 10, 40, 0.005, 0.01, skew_bound / 100, skew_bound / 10};
    auto suffix = "skew_" + std::to_string(skew_bound);
    // drop "0" in the suffix end
    while (suffix.back() == '0') {
      suffix.pop_back();
    }

    auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file/" + suffix;

    tree_builder.runIterativeFixSkewTest(env_info, case_num, skew_bound, dir, suffix);
  });
}

}  // namespace