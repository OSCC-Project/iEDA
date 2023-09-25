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
 * @file SolverTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include <vector>

#include "SolverAux.hh"
#include "gtest/gtest.h"

namespace {
class SolverTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(SolverTest, GeomTest)
{
  using icts::bst::GeomCalc;
  using icts::bst::Pt;
  std::vector<Pt> poly = {Pt(0, 0), Pt(0, 0.5), Pt(0, 1), Pt(1, 1), Pt(1, 0)};
  GeomCalc::convexHull(poly);
  EXPECT_EQ(poly.size(), 4);
}

TEST_F(SolverTest, SimpleTreeBuilderTest)
{
  TreeBuilderTest tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                               "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  double skew_bound = 0.08;
  tree_builder.runFixedTest(skew_bound);
}

TEST_F(SolverTest, RegressionTreeBuilderTest)
{
  TreeBuilderTest tree_builder("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                               "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  double skew_bound = 0.01;
  size_t case_num = 500;
  // design DB unit is 2000
  EnvInfo env_info{50000, 200000, 50000, 200000, 20, 40};
  auto data_set = tree_builder.runRegressTest(env_info, case_num, skew_bound);

  auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/file";
  auto method_list = {TreeBuilder::funcName(TreeBuilder::fluteTree), TreeBuilder::funcName(TreeBuilder::shallowLightTree),
                      TreeBuilder::funcName(TreeBuilder::boundSkewTree), TreeBuilder::funcName(TreeBuilder::bstSaltTree),
                      TreeBuilder::funcName(TreeBuilder::beatTree)};
  auto topo_type_list = {
      TopoTypeToString(TopoType::kGreedyDist),
      TopoTypeToString(TopoType::kGreedyMerge),
      TopoTypeToString(TopoType::kBiCluster),
      TopoTypeToString(TopoType::kBiPartition),
  };
  data_set.writeCSV(method_list, topo_type_list, dir, "regression.csv");

  auto target_method = TreeBuilder::funcName(TreeBuilder::beatTree);
  std::ranges::for_each(topo_type_list, [&](const std::string& topo_type) {
    auto ref_method = TreeBuilder::funcName(TreeBuilder::fluteTree);
    data_set.writeReduceCSV(target_method, ref_method, topo_type, dir);
    ref_method = TreeBuilder::funcName(TreeBuilder::shallowLightTree);
    data_set.writeReduceCSV(target_method, ref_method, topo_type, dir);
    ref_method = TreeBuilder::funcName(TreeBuilder::boundSkewTree);
    data_set.writeReduceCSV(target_method, ref_method, topo_type, dir);
    ref_method = TreeBuilder::funcName(TreeBuilder::bstSaltTree);
    data_set.writeReduceCSV(target_method, ref_method, topo_type, dir);
  });
}

}  // namespace