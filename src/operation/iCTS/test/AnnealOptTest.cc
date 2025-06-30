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
 * @file AnnealOptTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include <gtest/gtest.h>

#include "TestInterface.hh"
#include "anneal_opt/AnnealOpt.hh"
#include "balance_clustering/BalanceClustering.hh"
#include "log/Log.hh"
namespace {
using icts::BalanceClustering;
using icts::LatAnnealOpt;
using icts::VioAnnealOpt;

class AnnealOptAux : public TestInterface
{
 public:
  AnnealOptAux(const std::string& db_config_path, const std::string& cts_config_path) : TestInterface(db_config_path, cts_config_path)
  {
    LOG_INFO << "Anneal Optimize Test";
  }

  void runLatencyCostTest(const EnvInfo& env_info, const size_t& cluster_num, const size_t& max_iter, const double& cooling_rate,
                          const double& temperature) const
  {
    auto bufs = genRandomBuffers(env_info);
    // auto clusters = BalanceClustering::kMeansPlus(bufs, cluster_num);
    // LatAnnealOpt solver(clusters);
    // solver.initParameter(max_iter, cooling_rate, temperature);
    // clusters = solver.run(true);
    // std::ranges::for_each(bufs, [](auto& buf) { delete buf; });
  }

  void runViolationCostTest(const EnvInfo& env_info, const size_t& cluster_num, const size_t& max_iter, const double& cooling_rate,
                            const double& temperature, const int& max_fanout, const double& max_cap, const double& max_net_len,
                            const double& skew_bound) const
  {
    auto bufs = genRandomBuffers(env_info);
    // auto clusters = BalanceClustering::kMeansPlus(bufs, cluster_num);
    // BalanceClustering::writeClusterPy(clusters, "before");
    // VioAnnealOpt solver(clusters);
    // solver.initParameter(max_iter, cooling_rate, temperature, max_fanout, max_cap, max_net_len, skew_bound);
    // clusters = solver.run(true);
    // BalanceClustering::writeClusterPy(clusters, "after");
    // std::ranges::for_each(bufs, [](auto& buf) { delete buf; });
  }

 private:
};

class AnnealOptTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "AnnealOptTest";
    char* argv[] = {config};
    ieda::Log::init(argv);
  }
  void TearDown() { ieda::Log::end(); }
};

TEST_F(AnnealOptTest, LatencyCostTest)
{
  AnnealOptAux anneal_opt("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                          "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  EnvInfo env_info{50000, 500000, 50000, 500000, 400, 450};
  size_t cluster_num = 10;
  size_t max_iter = 200;
  double cooling_rate = 0.99;
  double temperature = 8000.0;
  anneal_opt.runLatencyCostTest(env_info, cluster_num, max_iter, cooling_rate, temperature);
}

TEST_F(AnnealOptTest, ViolationCostSmallTest)
{
  AnnealOptAux anneal_opt("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                          "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  EnvInfo env_info{50000, 500000, 50000, 500000, 270, 320};
  size_t cluster_num = 12;
  size_t max_iter = 200;
  double cooling_rate = 0.99;
  double temperature = 50000.0;
  auto max_fanout = TimingPropagator::getMaxFanout();
  auto max_cap = TimingPropagator::getMaxCap();
  auto max_net_len = TimingPropagator::getMaxLength();
  auto skew_bound = TimingPropagator::getSkewBound();
  anneal_opt.runViolationCostTest(env_info, cluster_num, max_iter, cooling_rate, temperature, max_fanout, max_cap, max_net_len, skew_bound);
}

TEST_F(AnnealOptTest, ViolationCostTest)
{
  AnnealOptAux anneal_opt("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json",
                          "/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  EnvInfo env_info{50000, 1500000, 50000, 1500000, 7500, 8000};
  size_t cluster_num = 250;
  size_t max_iter = 200;
  double cooling_rate = 0.99;
  double temperature = 60000.0;
  auto max_fanout = TimingPropagator::getMaxFanout();
  auto max_cap = TimingPropagator::getMaxCap();
  auto max_net_len = TimingPropagator::getMaxLength();
  auto skew_bound = TimingPropagator::getSkewBound();
  anneal_opt.runViolationCostTest(env_info, cluster_num, max_iter, cooling_rate, temperature, max_fanout, max_cap, max_net_len, skew_bound);
}
}  // namespace