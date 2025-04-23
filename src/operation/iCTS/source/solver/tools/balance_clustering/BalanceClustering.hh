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
#include <limits>

#include "Inst.hh"

namespace icts {
enum class EnhanceType
{
  kMinDelay,
  kMaxDelay,
  kWorstViolation,
};
struct ViolationScore
{
  double skew_vio_score;
  double cap_vio_score;
  double net_len_vio_score;
  std::vector<Inst*> cluster;
};
class LCA
{
 public:
  LCA(Node* root) : _root(root) { init(); }

  ~LCA() = default;

  Node* query(const std::vector<Node*>& nodes)
  {
    // find min/max dfs id
    int min_dfs_id = std::numeric_limits<int>::max();
    int max_dfs_id = std::numeric_limits<int>::min();
    for (auto node : nodes) {
      min_dfs_id = std::min(min_dfs_id, _node_id[node]);
      max_dfs_id = std::max(max_dfs_id, _node_id[node]);
    }
    return query(_nodes[min_dfs_id], _nodes[max_dfs_id]);
  }

  Node* query(Node* node1, Node* node2)
  {
    int l = _node_id[node1];
    int r = _node_id[node2];
    if (l > r) {
      std::swap(l, r);
    }
    return _nodes[query(l, r)];
  }

 private:
  void init()
  {
    // dfs
    dfs(_root, 0);
    // init rmq
    int n = _nodes.size();
    int k = 0;
    while ((1 << k) < n) {
      k++;
    }
    _rmq.resize(n, std::vector<int>(k + 1));
    _log2.resize(n + 1);
    _log2[1] = 0;
    for (int i = 2; i <= n; i++) {
      _log2[i] = _log2[i / 2] + 1;
    }
    for (int i = 0; i < n; i++) {
      _rmq[i][0] = i;
    }
    for (int j = 1; j <= k; j++) {
      for (int i = 0; i + (1 << j) - 1 < n; i++) {
        int lca1 = _rmq[i][j - 1];
        int lca2 = _rmq[i + (1 << (j - 1))][j - 1];
        _rmq[i][j] = _depths[lca1] < _depths[lca2] ? lca1 : lca2;
      }
    }
  }

  void dfs(Node* node, int depth)
  {
    _nodes.push_back(node);
    _node_id[node] = _nodes.size() - 1;
    _depths.push_back(depth);
    for (auto child : node->get_children()) {
      dfs(child, depth + 1);
      _nodes.push_back(node);
      _node_id[node] = _nodes.size() - 1;
      _depths.push_back(depth);
    }
  }

  // find LCA by Four Russians Algorithm and +1/-1 RMQ technique
  int query(int l, int r)
  {
    int len = r - l + 1;
    int k = _log2[len];
    int lca1 = _rmq[l][k];
    int lca2 = _rmq[r - (1 << k) + 1][k];
    return _depths[lca1] < _depths[lca2] ? lca1 : lca2;
  }

  Node* _root;
  std::vector<Node*> _nodes;
  std::unordered_map<Node*, int> _node_id;
  std::vector<int> _depths;
  std::vector<std::vector<int>> _rmq;
  std::vector<int> _log2;
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
  static std::vector<std::vector<Inst*>> kMeansPlus(const std::vector<Inst*>& insts, const size_t& k, const int& seed = 0,
                                                    const size_t& max_iter = 100, const size_t& no_change_stop = 5);

  static std::vector<std::vector<Inst*>> kMeans(const std::vector<Inst*>& insts, const size_t& k, const int& seed = 0,
                                                const size_t& max_iter = 100);

  static std::vector<std::vector<Inst*>> iterClustering(const std::vector<Inst*>& insts, const size_t& max_fanout,
                                                        const size_t& iters = 100, const size_t& no_change_stop = 5,
                                                        const double& limit_ratio = 0.8, const bool& log = false);

  static std::vector<std::vector<Inst*>> slackClustering(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                                         const size_t& max_fanout);

  static std::vector<std::vector<Inst*>> clusteringEnhancement(const std::vector<std::vector<Inst*>>& clusters, const int& max_fanout,
                                                               const double& max_cap, const double& max_net_length,
                                                               const double& skew_bound, const size_t& max_iter = 200,
                                                               const double& cooling_rate = 0.99, const double& temperature = 50000);

  static std::vector<Point> guideCenter(const std::vector<std::vector<Inst*>>& clusters, const std::optional<Point>& center = std::nullopt,
                                        const double& min_length = 50, const size_t& level = 1);

  static std::vector<Inst*> getMinDelayCluster(const std::vector<std::vector<Inst*>>& clusters);

  static std::vector<Inst*> getMaxDelayCluster(const std::vector<std::vector<Inst*>>& clusters);

  static std::vector<Inst*> getWorstViolationCluster(const std::vector<std::vector<Inst*>>& clusters);

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

  static double estimateNetDelay(const std::vector<Inst*>& cluster, const bool& is_max = true);

  static double estimateNetCap(const std::vector<Inst*>& cluster);

  static double estimateNetLength(const std::vector<Inst*>& cluster);

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

  static ViolationScore calcScore(const std::vector<Inst*>& cluster);

  static double crossProduct(const Point& p1, const Point& p2, const Point& p3);

  static void convexHull(std::vector<Point>& pts);

  static std::vector<CtsPoint<double>> paretoFront(const std::vector<CtsPoint<double>>& pts);

  static bool isContain(const Point& p, const std::vector<Point>& pts);

  static bool isSame(const std::vector<std::vector<Inst*>>& clusters1, const std::vector<std::vector<Inst*>>& clusters2);

  static void writeClusterPy(const std::vector<std::vector<Inst*>>& clusters, const std::string& save_name = "clusters");
};
}  // namespace icts