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
 * @file BalanceClustering.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "BalanceClustering.hh"

#include <algorithm>
#include <limits>
#include <random>
#include <ranges>

#include "CTSAPI.hpp"
#include "CtsConfig.h"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "log/Log.hh"
#include "min_cost_flow/MinCostFlow.hh"
#include "mip/MIP.hh"
namespace icts {
/**
 * @brief init clustering
 *
 * @param insts
 * @param k
 * @param seed
 * @param max_iter
 * @param no_change_stop
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::kMeans(const std::vector<Inst*>& insts, const size_t& k, const int& seed,
                                                          const size_t& max_iter, const size_t& no_change_stop)
{
  std::vector<std::vector<Inst*>> best_clusters(k);

  std::vector<Point> centers;
  size_t num_instances = insts.size();
  std::vector<int> assignments(num_instances);

  // Randomly choose first center from instances
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_instances - 1);
  auto loc = insts[dis(gen)]->get_location();
  centers.emplace_back(Point(loc.x(), loc.y()));
  // Choose k-1 remaining centers using kmeans++ algorithm
  while (centers.size() < k) {
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(insts[i]->get_location(), centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }
    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    auto select_loc = insts[selected_index]->get_location();
    centers.emplace_back(Point(select_loc.x(), select_loc.y()));
  }

  size_t num_iterations = 0;
  double prev_cap_variance = std::numeric_limits<double>::max();
  size_t no_change = 0;
  while (num_iterations++ < max_iter && no_change++ < no_change_stop) {
    // Assignment step
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(insts[i]->get_location(), centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
    }
    // Update step
    std::vector<Point> new_centers(k, Point(0, 0));
    std::vector<int> center_counts(k, 0);
    for (size_t i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      new_centers[center_index] += insts[i]->get_location();
      center_counts[center_index]++;
    }
    for (size_t i = 0; i < k; i++) {
      if (center_counts[i] > 0) {
        new_centers[i] /= center_counts[i];
      }
    }
    centers = new_centers;
    // Check cap variance
    std::vector<double> cluster_cap(k, 0);
    for (size_t i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      cluster_cap[center_index] += insts[i]->getCapLoad();
    }
    double cap_variance = calcVariance(cluster_cap);
    // Check for convergence
    if (cap_variance < prev_cap_variance) {
      best_clusters.clear();
      best_clusters.resize(k);
      prev_cap_variance = cap_variance;
      for (size_t i = 0; i < num_instances; i++) {
        int center_index = assignments[i];
        best_clusters[center_index].push_back(insts[i]);
      }
      no_change = 0;
    }
  }
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Inst*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
/**
 * @brief iter clustering
 *
 * @param insts
 * @param max_fanout
 * @param iters
 * @param no_change_stop
 * @param limit_ratio
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::iterClustering(const std::vector<Inst*>& insts, const size_t& max_fanout,
                                                                  const size_t& iters, const size_t& no_change_stop,
                                                                  const double& limit_ratio)
{
  LOG_FATAL_IF(max_fanout < 2) << "max_fanout should be greater than 1";
  if (insts.size() == 2) {
    return {insts};
  }
  LOG_INFO << "iterative clustering";
  // initialize clusters
  size_t cluster_num = std::ceil(1.0 * insts.size() / (limit_ratio * max_fanout));
  if (cluster_num == insts.size()) {
    cluster_num = insts.size() - 1;
  }
  auto clusters = kMeans(insts, cluster_num);
  size_t kmeans_num
      = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const std::vector<Inst*>& c) { return sum + c.size(); });
  LOG_FATAL_IF(kmeans_num != insts.size()) << "num of insts is not equal to num of clusters";
  LOG_FATAL_IF(max_fanout * clusters.size() < insts.size()) << "multiply max_fanout by num of clusters is less than num of insts";
  LOG_INFO << "initial clusters: " << clusters.size();
  if (clusters.size() == 1) {
    return clusters;
  }
  // make temp centroid buffer
  std::vector<Point> buffers = getCentroidBuffers(clusters);
  auto best_clusters = clusters;
  auto kmeans_var = calcBalanceVariance(clusters, buffers);
  auto mcf_var = std::numeric_limits<double>::max();
  LOG_INFO << "initial kmeans var: " << kmeans_var;
  size_t no_change = 0;
  // iterate
  for (size_t i = 0; i < iters; ++i, ++no_change) {
    if (no_change > no_change_stop) {
      LOG_INFO << "no change should stop in iter: " << i;
      break;
    }
    if ((i + 1) % 50 == 0) {
      LOG_INFO << "iter: " << i + 1;
    }
    MinCostFlow<Inst*> mcf;
    std::ranges::for_each(insts, [&mcf](Inst* inst) { mcf.add_node(inst->get_location().x(), inst->get_location().y(), inst); });
    std::ranges::for_each(buffers, [&mcf](const Point& buffer) { mcf.add_center(buffer.x(), buffer.y()); });
    clusters = mcf.run(max_fanout);
    size_t mcf_num
        = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const std::vector<Inst*>& c) { return sum + c.size(); });
    LOG_FATAL_IF(mcf_num != insts.size()) << "num of insts is not equal to num of clusters";
    buffers = getCentroidBuffers(clusters);
    auto new_mcf_var = calcBalanceVariance(clusters, buffers);
    if (new_mcf_var < mcf_var) {
      mcf_var = new_mcf_var;
      best_clusters = clusters;
      no_change = 0;
      LOG_INFO << "update in mcf iter: " << i + 1;
    } else {
      clusters = kMeans(insts, cluster_num, i, 5);
      auto temp_buffers = getCentroidBuffers(clusters);
      auto temp_kmeans_var = calcBalanceVariance(clusters, temp_buffers);
      if (temp_kmeans_var < kmeans_var) {
        kmeans_var = temp_kmeans_var;
        buffers = temp_buffers;
        no_change = 0;
        LOG_INFO << "update in kmeans iter: " << i + 1;
      }
    }
  }
  LOG_INFO << "final mcf var: " << mcf_var;
  LOG_INFO << "final kmeans var: " << kmeans_var;
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Inst*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
/**
 * @brief slack net length clustering
 *
 * @param clusters
 * @param max_net_length
 * @param max_fanout
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::slackClustering(const std::vector<std::vector<Inst*>>& clusters,
                                                                   const double& max_net_length, const size_t& max_fanout)
{
  std::vector<std::vector<Inst*>> slack_clusters;
  std::ranges::for_each(clusters, [&](const std::vector<Inst*>& cluster) {
    auto est_net_length = estimateNetLength(cluster, max_net_length, max_fanout);
    if (est_net_length < max_net_length) {
      slack_clusters.push_back(cluster);
    } else {
      // reclustering
      auto new_clusters = kMeans(cluster, std::ceil(est_net_length / max_net_length));
      std::ranges::for_each(new_clusters, [&](const std::vector<Inst*>& new_cluster) {
        auto re_slack_clusters = slackClustering({new_cluster}, max_net_length, max_fanout);
        slack_clusters.insert(slack_clusters.end(), re_slack_clusters.begin(), re_slack_clusters.end());
      });
    }
  });
  return slack_clusters;
}
/**
 * @brief clustering enhancement by MIP
 *
 * @param clusters
 * @param max_fanout
 * @param max_cap
 * @param max_net_length
 * @param iter
 * @param p
 * @param q
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::clusteringEnhancement(const std::vector<std::vector<Inst*>>& clusters,
                                                                         const int& max_fanout, const double& max_cap,
                                                                         const double& max_net_length, const size_t& iter, const double& p,
                                                                         const double& q, const double& r)
{
  auto enhanced_clusters = clusters;
  auto back_up = enhanced_clusters;
  for (size_t i = 0; i < iter; ++i) {
    // // opt min delay cluster
    auto min_delay_cluster = getMinDelayCluster(enhanced_clusters, max_net_length, max_fanout);
    enhanced_clusters
        = mipEnhancement(enhanced_clusters, min_delay_cluster, max_fanout, max_cap, max_net_length, EnhanceType::kMIN_DELAY, p, q, r);
    // check result
    auto new_min_cluster = getMinDelayCluster(enhanced_clusters, max_net_length, max_fanout);
    auto new_min_delay = estimateNetDelay(new_min_cluster, max_net_length, max_fanout);
    auto old_min_delay = estimateNetDelay(min_delay_cluster, max_net_length, max_fanout);
    if (new_min_delay < old_min_delay) {
      enhanced_clusters = back_up;
    }
    // opt max delay cluster
    auto max_delay_cluster = getMaxDelayCluster(enhanced_clusters, max_net_length, max_fanout);
    enhanced_clusters
        = mipEnhancement(enhanced_clusters, max_delay_cluster, max_fanout, max_cap, max_net_length, EnhanceType::kMAX_DELAY, p, q, r);
    // check result
    auto new_max_cluster = getMaxDelayCluster(enhanced_clusters, max_net_length, max_fanout);
    auto new_max_delay = estimateNetDelay(new_max_cluster, max_net_length, max_fanout);
    auto old_max_delay = estimateNetDelay(max_delay_cluster, max_net_length, max_fanout);
    if (new_max_delay > old_max_delay) {
      enhanced_clusters = back_up;
    }
    // screen violation cluster
    auto worst_cluster = getWorstViolationCluster(enhanced_clusters, max_cap, max_net_length, max_fanout);
    if (worst_cluster.empty()) {
      if (isSame(enhanced_clusters, back_up)) {
        break;
      }
      back_up = enhanced_clusters;
      continue;
    }
    enhanced_clusters
        = mipEnhancement(enhanced_clusters, worst_cluster, max_fanout, max_cap, max_net_length, EnhanceType::kWORST_VIOLATION, p, q, r);
    auto new_worst_cluster = getWorstViolationCluster(enhanced_clusters, max_cap, max_net_length, max_fanout);
    if (!new_worst_cluster.empty()) {
      auto new_skew = estimateSkew(new_worst_cluster);
      auto old_skew = estimateSkew(worst_cluster);
      if (new_skew > old_skew) {
        enhanced_clusters = back_up;
      }
    }
    // check no change
    if (isSame(enhanced_clusters, back_up)) {
      break;
    }
    back_up = enhanced_clusters;
  }
  return enhanced_clusters;
}
/**
 * @brief MIP enhancement
 *
 * @param enhanced_clusters
 * @param center_cluster
 * @param max_fanout
 * @param max_cap
 * @param max_net_length
 * @param enhance_type
 * @param p
 * @param q
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::mipEnhancement(std::vector<std::vector<Inst*>>& enhanced_clusters,
                                                                  const std::vector<Inst*>& center_cluster, const int& max_fanout,
                                                                  const double& max_cap, const double& max_net_length,
                                                                  const EnhanceType& enhance_type, const double& p, const double& q,
                                                                  const double& r)
{
  // remove center cluster
  enhanced_clusters.erase(std::remove_if(enhanced_clusters.begin(), enhanced_clusters.end(),
                                         [&center_cluster](const std::vector<Inst*>& cluster) { return cluster == center_cluster; }),
                          enhanced_clusters.end());
  // find the closest clusters
  std::vector<std::vector<Inst*>> neighbors = {};

  neighbors = getMostRecentClusters(enhanced_clusters, center_cluster);

  // remove neighbors
  enhanced_clusters.erase(std::remove_if(enhanced_clusters.begin(), enhanced_clusters.end(),
                                         [&neighbors](const std::vector<Inst*>& cluster) {
                                           return std::find(neighbors.begin(), neighbors.end(), cluster) != neighbors.end();
                                         }),
                          enhanced_clusters.end());
  // enhanced center and neighbors
  auto to_be_enhanced_clusters = neighbors;
  to_be_enhanced_clusters.push_back(center_cluster);
  // writeClusterPy(to_be_enhanced_clusters, "mip_before");
  auto mip_solver = MIP(to_be_enhanced_clusters);
  size_t net_num = to_be_enhanced_clusters.size();
  mip_solver.initParameter(net_num, max_fanout, max_cap, max_net_length * TimingPropagator::getDbUnit(), p, q, r);
  auto opt_clusters = mip_solver.run();
  while (opt_clusters.empty()) {
    std::ranges::copy(neighbors, std::back_inserter(enhanced_clusters));
    neighbors = getMostRecentClusters(enhanced_clusters, center_cluster, 36, 2);
    enhanced_clusters.erase(std::remove_if(enhanced_clusters.begin(), enhanced_clusters.end(),
                                           [&neighbors](const std::vector<Inst*>& cluster) {
                                             return std::find(neighbors.begin(), neighbors.end(), cluster) != neighbors.end();
                                           }),
                            enhanced_clusters.end());
    to_be_enhanced_clusters = neighbors;
    to_be_enhanced_clusters.push_back(center_cluster);
    // writeClusterPy(to_be_enhanced_clusters, "mip_before");
    auto temp_solver = MIP(to_be_enhanced_clusters);
    temp_solver.initParameter(net_num, max_fanout, max_cap, max_net_length * TimingPropagator::getDbUnit(), p, q, r);
    opt_clusters = temp_solver.run();
    net_num += 1;
  }
  opt_clusters.erase(
      std::remove_if(opt_clusters.begin(), opt_clusters.end(), [](const std::vector<Inst*>& cluster) { return cluster.empty(); }),
      opt_clusters.end());
  // writeClusterPy(opt_clusters, "mip_after");
  std::ranges::for_each(opt_clusters, [&enhanced_clusters](const std::vector<Inst*>& cluster) { enhanced_clusters.push_back(cluster); });
  return enhanced_clusters;
}
/**
 * @brief get the min delay cluster
 *
 * @param clusters
 * @param max_net_length
 * @param max_fanout
 * @return std::vector<Inst*>
 */
std::vector<Inst*> BalanceClustering::getMinDelayCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                                         const size_t& max_fanout)
{
  std::map<size_t, double> delay_map;
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    delay_map[i] = estimateNetDelay(cluster, max_net_length, max_fanout, false);
  }
  auto min_delay_cluster = clusters[std::min_element(delay_map.begin(), delay_map.end(), [](const auto& a, const auto& b) {
                                      return a.second < b.second;
                                    })->first];
  return min_delay_cluster;
}
/**
 * @brief get the max delay cluster
 *
 * @param clusters
 * @param max_net_length
 * @param max_fanout
 * @return std::vector<Inst*>
 */
std::vector<Inst*> BalanceClustering::getMaxDelayCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_net_length,
                                                         const size_t& max_fanout)
{
  std::map<size_t, double> delay_map;
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    delay_map[i] = estimateNetDelay(cluster, max_net_length, max_fanout);
  }
  auto max_delay_cluster = clusters[std::max_element(delay_map.begin(), delay_map.end(), [](const auto& a, const auto& b) {
                                      return a.second < b.second;
                                    })->first];
  return max_delay_cluster;
}
/**
 * @brief get worst violation cluster
 *
 * @param clusters
 * @param max_cap
 * @param max_net_length
 * @param max_fanout
 * @return std::vector<Inst*>
 */
std::vector<Inst*> BalanceClustering::getWorstViolationCluster(const std::vector<std::vector<Inst*>>& clusters, const double& max_cap,
                                                               const double& max_net_length, const size_t& max_fanout)
{
  // sort violation by skew, then cap, then net length
  auto vio_cmp = [](const ViolationScore& a, const ViolationScore& b) {
    return a.skew_vio_score > b.skew_vio_score || (a.skew_vio_score == b.skew_vio_score && a.cap_vio_score > b.cap_vio_score)
           || (a.skew_vio_score == b.skew_vio_score && a.cap_vio_score == b.cap_vio_score && a.net_len_vio_score > b.net_len_vio_score);
  };
  std::vector<ViolationScore> vio_scores;
  std::ranges::for_each(clusters, [&vio_scores, &max_cap, &max_net_length, &max_fanout](const std::vector<Inst*>& cluster) {
    auto score = calcScore(cluster, max_cap, max_net_length, max_fanout);
    vio_scores.push_back(score);
  });
  std::sort(vio_scores.begin(), vio_scores.end(), vio_cmp);
  auto is_violation = [&](const ViolationScore& vio_score) {
    if (vio_score.skew_vio_score > TimingPropagator::getSkewBound() || vio_score.cap_vio_score > max_cap
        || vio_score.net_len_vio_score > max_net_length) {
      return true;
    }
    return false;
  };
  for (auto& vio_score : vio_scores) {
    if (is_violation(vio_score)) {
      return vio_score.cluster;
    }
  }
  return {};
}
/**
 * @brief get most recent clusters
 *
 * @param clusters
 * @param center_cluster
 * @param num_limit
 * @param cluster_num_limit
 * @return std::vector<std::vector<Inst*>>
 */
std::vector<std::vector<Inst*>> BalanceClustering::getMostRecentClusters(const std::vector<std::vector<Inst*>>& clusters,
                                                                         const std::vector<Inst*>& center_cluster, const size_t& num_limit,
                                                                         const size_t& cluster_num_limit)
{
  std::vector<std::pair<size_t, double>> id_dist_pairs;
  auto loc = calcCentroid(center_cluster);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto center = calcCentroid(clusters[i]);
    auto dist = TimingPropagator::calcDist(loc, center);
    id_dist_pairs.push_back({i, dist});
  }
  std::sort(id_dist_pairs.begin(), id_dist_pairs.end(),
            [](const std::pair<size_t, double>& p1, const std::pair<size_t, double>& p2) { return p1.second < p2.second; });
  std::vector<std::vector<Inst*>> recent_clusters;
  size_t inst_num = center_cluster.size();
  for (auto id_dist_pair : id_dist_pairs) {
    auto id = id_dist_pair.first;
    auto cluster = clusters[id];
    inst_num += cluster.size();
    if (inst_num > num_limit && !recent_clusters.empty()) {
      break;
    }
    recent_clusters.push_back(cluster);
    if (recent_clusters.size() == cluster_num_limit) {
      break;
    }
  }
  return recent_clusters;
}
/**
 * @brief generate centroid buffers location
 *
 * @param clusters
 * @return std::vector<Point>
 */
std::vector<Point> BalanceClustering::getCentroidBuffers(const std::vector<std::vector<Inst*>>& clusters)
{
  std::vector<Point> buffers;
  std::ranges::for_each(clusters, [&buffers](const std::vector<Inst*>& cluster) {
    auto centroid = calcCentroid(cluster);
    buffers.push_back(centroid);
  });
  return buffers;
}
/**
 * @brief divide insts by function to pair<lower: num * ratio, higher: num * (1 - ratio)>
 *
 * @param insts
 * @return std::pair<std::vector<Inst*>, std::vector<Inst*>>
 */
std::pair<std::vector<Inst*>, std::vector<Inst*>> BalanceClustering::divideBy(const std::vector<Inst*>& insts,
                                                                              const std::function<double(const Inst*)>& func,
                                                                              const double& ratio)
{
  auto cmp = [&](const Inst* inst1, const Inst* inst2) { return func(inst1) < func(inst2); };
  std::vector<Inst*> sorted_insts = insts;
  std::sort(sorted_insts.begin(), sorted_insts.end(), cmp);
  size_t num = sorted_insts.size();
  size_t split_id = num * ratio;
  std::vector<Inst*> left(sorted_insts.begin(), sorted_insts.begin() + split_id);
  std::vector<Inst*> right(sorted_insts.begin() + split_id, sorted_insts.end());
  return {left, right};
}
/**
 * @brief get the clusters on the boundary
 *
 * @param clusters
 * @return std::pair<std::vector<std::vector<Inst*>>, std::vector<std::vector<Inst*>>>
 */
std::pair<std::vector<std::vector<Inst*>>, std::vector<std::vector<Inst*>>> BalanceClustering::getBoundCluster(
    const std::vector<std::vector<Inst*>>& clusters)
{
  auto centers = getCentroidBuffers(clusters);

  auto convex = centers;
  pgl::convex_hull(convex);
  // Filter all clusters where center is on the convex package
  auto remain_centers = centers;
  auto is_in_convex = [&convex](const Point& center) { return std::find(convex.begin(), convex.end(), center) != convex.end(); };
  remain_centers.erase(
      std::remove_if(remain_centers.begin(), remain_centers.end(), [&](const Point& center) { return is_in_convex(center); }),
      remain_centers.end());
  auto second_convex = remain_centers;
  if (!second_convex.empty()) {
    pgl::convex_hull(second_convex);
  }
  convex.insert(convex.end(), second_convex.begin(), second_convex.end());

  std::vector<std::vector<Inst*>> bound_clusters;
  std::vector<std::vector<Inst*>> remain_clusters;
  std::ranges::for_each(clusters, [&](const std::vector<Inst*>& cluster) {
    auto center = calcCentroid(cluster);
    if (is_in_convex(center)) {
      bound_clusters.push_back(cluster);
    } else {
      remain_clusters.push_back(cluster);
    }
  });
  LOG_FATAL_IF(bound_clusters.size() + remain_clusters.size() != clusters.size())
      << "cluster size error, bound clusters size: " << bound_clusters.size() << ", remain clusters size: " << remain_clusters.size()
      << ", clusters size: " << clusters.size();
  return {bound_clusters, remain_clusters};
}
/**
 * @brief optimize latency of cluster
 *
 * @param cluster
 * @param skew_bound
 * @param ratio
 */
void BalanceClustering::latencyOpt(const std::vector<Inst*> cluster, const double& skew_bound, const double& ratio)
{
  if (cluster.size() * ratio < 1 || cluster.size() * (1 - ratio) < 1) {
    return;
  }
  auto get_max_delay = [&](const Inst* inst) {
    if (inst->isSink()) {
      return 0.0;
    }
    TimingPropagator::initLoadPinDelay(inst->get_load_pin(), true);
    return inst->get_load_pin()->get_max_delay();
  };
  auto divide = divideBy(cluster, get_max_delay, ratio);
  auto feasible_insts = divide.first;
  auto delay_bound = feasible_insts.back()->get_load_pin()->get_max_delay();
  auto to_be_amplify = divide.second;
  std::ranges::for_each(to_be_amplify, [&](Inst* inst) {
    auto* driver_pin = inst->get_driver_pin();
    auto* load_pin = inst->get_load_pin();
    auto* net = driver_pin->get_net();
    auto feasible_cell = TreeBuilder::feasibleCell(inst, skew_bound);
    for (auto cell : feasible_cell) {
      inst->set_cell_master(cell);
      TimingPropagator::update(net);
      TimingPropagator::initLoadPinDelay(load_pin, true);
      auto max_delay = load_pin->get_max_delay();
      if (max_delay < delay_bound) {
        break;
      }
    }
  });
}
/**
 * @brief estimate skew of cluster
 *
 * @param cluster
 * @return double
 */
double BalanceClustering::estimateSkew(const std::vector<Inst*>& cluster)
{
  auto center = calcCentroid(cluster);
  auto buffer = TreeBuilder::genBufInst("temp", center);
  std::vector<Pin*> cluster_load_pins;
  std::ranges::for_each(cluster, [&cluster_load_pins](Inst* inst) {
    auto load_pin = inst->get_load_pin();
    cluster_load_pins.push_back(load_pin);
  });
  // set cell master
  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  // location legitimization
  TreeBuilder::localPlace(buffer, cluster_load_pins);
  auto* driver_pin = buffer->get_driver_pin();
  if (cluster_load_pins.size() == 1) {
    auto load_pin = cluster_load_pins.front();
    TreeBuilder::directConnectTree(driver_pin, load_pin);
  } else {
    TreeBuilder::shallowLightTree(driver_pin, cluster_load_pins);
  }
  auto* net = TimingPropagator::genNet("temp", driver_pin, cluster_load_pins);
  TimingPropagator::update(net);
  auto skew = driver_pin->get_max_delay() - driver_pin->get_min_delay();

  TreeBuilder::recoverNet(net);

  return skew;
}
/**
 * @brief estimate net delay of cluster
 *
 * @param cluster
 * @param max_net_length
 * @param max_fanout
 * @param is_max
 * @return double
 */
double BalanceClustering::estimateNetDelay(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout,
                                           const bool& is_max)
{
  auto net_len = estimateNetLength(cluster, max_net_length, max_fanout);
  auto total_cap = estimateNetCap(cluster, max_net_length, max_fanout);
  double res = net_len * TimingPropagator::getUnitRes();
  auto net_delay = total_cap * res / 2;
  auto target_delay = is_max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
  std::ranges::for_each(cluster, [&net_delay, &target_delay, &is_max](const Inst* inst) {
    auto cur_delay = (inst->isBuffer() ? inst->get_driver_pin()->get_max_delay() : 0) + net_delay;
    target_delay = is_max ? std::max(target_delay, cur_delay) : std::min(target_delay, cur_delay);
  });
  return target_delay;
}
/**
 * @brief  estimate cap of cluster
 *
 * @param cluster
 * @param max_net_length
 * @param max_fanout
 * @return double
 */
double BalanceClustering::estimateNetCap(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout)
{
  auto net_len = estimateNetLength(cluster, max_net_length, max_fanout);
  auto pin_cap
      = std::accumulate(cluster.begin(), cluster.end(), 0.0, [](double total, const Inst* inst) { return total + inst->getCapLoad(); });
  auto total_cap = net_len * TimingPropagator::getUnitCap() + pin_cap;
  return total_cap;
}
/**
 * @brief estimate net length of cluster
 *
 * @param cluster
 * @param max_net_length
 * @param max_fanout
 * @return double
 */
double BalanceClustering::estimateNetLength(const std::vector<Inst*>& cluster, const double& max_net_length, const size_t& max_fanout)
{
  auto hp_wl = calcHPWL(cluster);
  auto fanout = cluster.size();
  auto stn_wl = 1.3 * hp_wl + 2.53 * fanout;
  return stn_wl;
}
/**
 * @brief calculate centroid of cluster
 *
 * @param cluster
 * @return Point
 */
Point BalanceClustering::calcCentroid(const std::vector<Inst*>& cluster)
{
  auto centroid = CtsPoint<int64_t>(0, 0);
  centroid = std::accumulate(cluster.begin(), cluster.end(), centroid,
                             [](const CtsPoint<int64_t>& a, const Inst* b) { return a + b->get_location(); });
  centroid /= cluster.size();
  return centroid;
}
/**
 * @brief calculate bound centroid of cluster
 *
 * @param cluster
 * @return Point
 */
Point BalanceClustering::calcBoundCentroid(const std::vector<Inst*>& cluster)
{
  auto min_x = std::numeric_limits<int>::max();
  auto min_y = std::numeric_limits<int>::max();
  auto max_x = std::numeric_limits<int>::min();
  auto max_y = std::numeric_limits<int>::min();
  std::ranges::for_each(cluster, [&min_x, &min_y, &max_x, &max_y](const Inst* inst) {
    auto loc = inst->get_location();
    min_x = std::min(min_x, loc.x());
    min_y = std::min(min_y, loc.y());
    max_x = std::max(max_x, loc.x());
    max_y = std::max(max_y, loc.y());
  });
  return Point((min_x + max_x) / 2, (min_y + max_y) / 2);
}
/**
 * @brief calculate half-perimeter manhattan distance of cluster
 *
 * @param cluster
 * @return int
 */
int BalanceClustering::calcHPMD(const std::vector<Inst*>& cluster)
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  std::ranges::for_each(cluster, [&min_x, &min_y, &max_x, &max_y](const Inst* inst) {
    auto loc = inst->get_location();
    min_x = std::min(min_x, loc.x());
    min_y = std::min(min_y, loc.y());
    max_x = std::max(max_x, loc.x());
    max_y = std::max(max_y, loc.y());
  });
  return max_x - min_x + max_y - min_y;
}
/**
 * @brief calculate half-perimeter wire length of cluster
 *
 * @param cluster
 * @return double
 */
double BalanceClustering::calcHPWL(const std::vector<Inst*>& cluster)
{
  auto hpmd = calcHPMD(cluster);
  return 1.0 * hpmd / TimingPropagator::getDbUnit();
}
/**
 * @brief calculate sum of absolute error
 *
 * @param clusters
 * @param buffers
 * @return double
 */
double BalanceClustering::calcSAE(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers)
{
  double sae = 0;
  for (size_t i = 0; i < clusters.size(); ++i) {
    sae += std::accumulate(clusters[i].begin(), clusters[i].end(), 0.0, [&buffers, &i](double total, const Inst* inst) {
      return total + pgl::manhattan_distance(inst->get_location(), buffers[i]);
    });
  }
  return sae;
}
/**
 * @brief trade off between cap and delay
 *
 * @param clusters
 * @param buffers
 * @param cap_coef
 * @param delay_coef
 * @return double
 */
double BalanceClustering::calcBalanceVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers,
                                              const double& cap_coef, const double& delay_coef)
{
  return cap_coef * calcCapVariance(clusters, buffers) + delay_coef * calcDelayVariance(clusters, buffers);
}
/**
 * @brief calculate cap variance
 *
 * @param clusters
 * @param buffers
 * @return double
 */
double BalanceClustering::calcCapVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers)
{
  // cap variance
  std::vector<double> cluster_cap(clusters.size(), 0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    auto max_net_length = TimingPropagator::getMaxLength();
    auto max_fanout = TimingPropagator::getMaxFanout();
    cluster_cap[i] = estimateNetCap(cluster, max_net_length, max_fanout);
  }
  // unify cluster cap
  // auto max_val = std::max_element(std::begin(cluster_cap), std::end(cluster_cap));
  // std::ranges::for_each(cluster_cap, [&max_val](double& val) { val /= *max_val; });
  double cap_variance = calcVariance(cluster_cap);
  return cap_variance;
}
/**
 * @brief calculate delay variance
 *
 * @param clusters
 * @param buffers
 * @return double
 */
double BalanceClustering::calcDelayVariance(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point>& buffers)
{
  // delay variance
  std::vector<double> cluster_delay(clusters.size(), 0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    auto max_net_length = TimingPropagator::getMaxLength();
    auto max_fanout = TimingPropagator::getMaxFanout();
    cluster_delay[i] = estimateNetDelay(cluster, max_net_length, max_fanout);
  }
  // unify cluster delay
  // auto max_val = std::max_element(std::begin(cluster_delay), std::end(cluster_delay));
  // std::ranges::for_each(cluster_delay, [&max_val](double& val) { val /= *max_val; });
  double delay_variance = calcVariance(cluster_delay);
  return delay_variance;
}
/**
 * @brief calculate variance of values
 *
 * @param values
 * @return double
 */
double BalanceClustering::calcVariance(const std::vector<double>& values)
{
  double sum = std::accumulate(std::begin(values), std::end(values), 0.0);
  double mean = sum / values.size();
  double variance = std::accumulate(std::begin(values), std::end(values), 0.0,
                                    [&mean](double total, const double& val) { return total + std::pow(val - mean, 2); });
  variance /= values.size();
  return variance;
}
/**
 * @brief calculate score of cluster
 *
 * @param cluster
 * @param max_cap
 * @param max_net_length
 * @param max_fanout
 * @return ViolationScore
 */
ViolationScore BalanceClustering::calcScore(const std::vector<Inst*>& cluster, const double& max_cap, const double& max_net_length,
                                            const size_t& max_fanout)
{
  auto skew_vio_score = estimateSkew(cluster);
  auto cap_vio_score = estimateNetCap(cluster, max_net_length, max_fanout);
  auto net_len_vio_score = estimateNetLength(cluster, max_net_length, max_fanout);
  return ViolationScore{skew_vio_score, cap_vio_score, net_len_vio_score, cluster};
}
/**
 * @brief judge whether two clusters are the same
 *
 * @param clusters1
 * @param clusters2
 * @return true
 * @return false
 */
bool BalanceClustering::isSame(const std::vector<std::vector<Inst*>>& clusters1, const std::vector<std::vector<Inst*>>& clusters2)
{
  if (clusters1.size() != clusters2.size()) {
    return false;
  }
  for (auto cluster : clusters1) {
    if (std::find(clusters2.begin(), clusters2.end(), cluster) == clusters2.end()) {
      return false;
    }
  }
  return true;
}
/**
 * @brief write cluster to python file
 *
 * @param clusters
 * @param save_name
 */
void BalanceClustering::writeClusterPy(const std::vector<std::vector<Inst*>>& clusters, const std::string& save_name)
{
  LOG_INFO << "Writing clusters to python file...";
  // write the cluster to python file
  auto* config = CTSAPIInst.get_config();
  auto path = config->get_sta_workspace();
  std::ofstream ofs(path + "/" + save_name + ".py");
  ofs << "import matplotlib.pyplot as plt" << std::endl;
  ofs << "import numpy as np" << std::endl;
  ofs << "def generate_color_sequence(n): " << std::endl;
  ofs << "    cmap = plt.get_cmap('viridis')" << std::endl;
  ofs << "    colors = [cmap(i) for i in np.linspace(0, 1, n)]" << std::endl;
  ofs << "    return colors" << std::endl;
  ofs << "colors = generate_color_sequence(" << clusters.size() << ")" << std::endl;
  ofs << "fig = plt.figure(figsize=(8,6), dpi=300)" << std::endl;
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (clusters[i].empty()) {
      continue;
    }
    ofs << "x" << i << " = [";
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      ofs << clusters[i][j]->get_location().x();
      if (j != clusters[i].size() - 1) {
        ofs << ", ";
      }
    }
    ofs << "]" << std::endl;
    ofs << "y" << i << " = [";
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      ofs << clusters[i][j]->get_location().y();
      if (j != clusters[i].size() - 1) {
        ofs << ", ";
      }
    }
    ofs << "]" << std::endl;
    ofs << "plt.scatter(x" << i << ", y" << i << ",label='Class " + std::to_string(i + 1) + "',color=colors[" << i << "],s=1)" << std::endl;
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      ofs << "plt.text(" << clusters[i][j]->get_location().x() << ", " << clusters[i][j]->get_location().y() << ", '"
          << clusters[i][j]->getCapLoad() << "', fontsize=3)" << std::endl;
    }
    ofs << "rect" << i << " = plt.Rectangle((np.min(x" << i << "), np.min(y" << i << ")),"
        << "np.max(x" << i << ") - np.min(x" << i << "),"
        << "np.max(y" << i << ") - np.min(y" << i << "),"
        << "fill = False, edgecolor = colors[" << i << "])" << std::endl;
    ofs << "plt.gca().add_patch(rect" << i << ")" << std::endl;
  }
  ofs << "plt.subplots_adjust(right=0.8)" << std::endl;
  ofs << "plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)" << std::endl;
  ofs << "plt.axis('equal')" << std::endl;
  ofs << "plt.show()" << std::endl;
  ofs << "plt.savefig('" + save_name + ".png')" << std::endl;
  ofs.close();
}

}  // namespace icts