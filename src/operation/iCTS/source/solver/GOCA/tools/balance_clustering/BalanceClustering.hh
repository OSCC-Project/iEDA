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
 * @file BalanceClustering.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <vector>

#include "Inst.hh"

namespace icts {
enum class EnhanceType
{
  kMIN_DELAY,
  kMAX_DELAY,
  kWORST_VIOLATION,
};
struct ViolationScore
{
  double skew_vio_score;
  double cap_vio_score;
  double net_len_vio_score;
  std::vector<Inst*> cluster;
};
/**
 * @brief BalanceClustering class
 *       clustering sinks by max distance and max fanout
 *       using k-means algorithm construct a initial clustering
 *       then iteratively adjust the clustering by min cost flow to balance the capacitance variance
 *
 */
class BalanceClustering
{
 public:
  BalanceClustering() = delete;
  ~BalanceClustering() = default;
  static std::vector<std::vector<Inst*>> kMeans(const std::vector<Inst*>& sinks, const size_t& k, const int& seed = 0,
                                                const size_t& max_iter = 100, const size_t& no_change_stop = 5);

  static std::vector<std::vector<Inst*>> iterClustering(const std::vector<Inst*>& sinks, const size_t& max_fanout,
                                                        const size_t& iters = 100, const size_t& no_change_stop = 5,
                                                        const double& limit_ratio = 0.8);

  static std::vector<std::vector<Inst*>> slackClustering(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                                         const size_t& max_fanout);

  static std::vector<std::vector<Inst*>> clusteringEnhancement(const std::vector<std::vector<Inst*>>& clusters, const int& max_fanout,
                                                               const double& max_cap, const double& max_net_length, const size_t& iter = 5,
                                                               const double& p = 5e-4, const double& q = 0.5, const double& r = 5000);

  static std::vector<std::vector<Inst*>> mipEnhancement(std::vector<std::vector<Inst*>>& enhanced_clusters,
                                                        const std::vector<Inst*>& center_cluster, const int& max_fanout,
                                                        const double& max_cap, const double& max_net_length,
                                                        const EnhanceType& enhance_type, const double& p, const double& q, const double& r);

  static std::vector<Inst*> getMinDelayCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                               const size_t& max_fanout);

  static std::vector<Inst*> getMaxDelayCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                               const size_t& max_fanout);

  static std::vector<Inst*> getWorstViolationCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_cap,
                                                     const double& max_net_length, const size_t& max_fanout);

  static std::vector<std::vector<Inst*>> getMostRecentClusters(const std::vector<std::vector<Inst*>>& clusters,
                                                               const std::vector<Inst*>& center_cluster, const size_t& num_limit = 42,
                                                               const size_t& cluster_num_limit = 4);

  static std::vector<Point> getCentroidBuffers(const std::vector<std::vector<Inst*>>& clusters);

  static std::pair<std::vector<Inst*>, std::vector<Inst*>> divideBy(const std::vector<Inst*>& insts,
                                                                    const std::function<double(const Inst*)>& func, const double& ratio);

  static std::pair<std::vector<std::vector<Inst*>>, std::vector<std::vector<Inst*>>> getBoundCluster(
      const std::vector<std::vector<Inst*>>& clusters);

  static void latencyOpt(const std::vector<Inst*> cluster, const double& skew_bound, const double& ratio);

  static double estimateSkew(const std::vector<Inst*>& cluster);

  static double estimateNetDelay(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout,
                                 const bool& is_max = true);

  static double estimateNetCap(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout);

  static double estimateNetLength(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout);

  static Point calcCentroid(const std::vector<Inst*>& cluster);

  static Point calcBoundCentroid(const std::vector<Inst*>& cluster);

  static int calcHPMD(const std::vector<Inst*>& cluster);

  static double calcHPWL(const std::vector<Inst*>& cluster);

  static double calcSAE(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers);

  static double calcBalanceVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers,
                                    const double& cap_coef = 1.0, const double& delay_coef = 1.0);

  static double calcCapVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers);

  static double calcDelayVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers);

  static double calcVariance(const std::vector<double>& values);

  static ViolationScore calcScore(const std::vector<Inst*>& cluster, const double& max_cap, const double& max_net_length,
                                  const size_t& max_fanout);

  static bool isSame(const std::vector<std::vector<Inst*>>& clusters1, const std::vector<std::vector<Inst*>>& clusters2);

  static void writeClusterPy(const std::vector<std::vector<Inst*>>& clusters, const std::string& save_name = "clusters");
};
}  // namespace icts