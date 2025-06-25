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
#include <filesystem>
#include <limits>
#include <random>
#include <ranges>

#include "CTSAPI.hh"
#include "CtsConfig.hh"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "anneal_opt/AnnealOpt.hh"
#include "log/Log.hh"
#include "min_cost_flow/MinCostFlow.hh"
namespace icts {
/**
 * @brief init clustering
 *
 * @param load_pins
 * @param k
 * @param seed
 * @param max_iter
 * @param no_change_stop
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::kMeansPlus(const std::vector<Pin*>& load_pins, const size_t& k, const int& seed,
                                                             const size_t& max_iter, const size_t& no_change_stop)
{
  std::vector<std::vector<Pin*>> best_clusters(k);

  std::vector<Point> centers;
  size_t num_pins = load_pins.size();
  std::vector<int> assignments(num_pins);

  // Randomly choose first center from load_pins
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_pins - 1);

  // Choose k-1 remaining centers using kmeans++ algorithm
  auto loc = load_pins[dis(gen)]->get_location();
  centers.emplace_back(loc);
  while (centers.size() < k) {
    std::vector<double> distances(num_pins, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_pins; i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = TimingPropagator::calcDist(load_pins[i]->get_location(), centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }
    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    auto select_loc = load_pins[selected_index]->get_location();
    centers.emplace_back(select_loc);
  }

  size_t num_iterations = 0;
  double prev_cap_variance = std::numeric_limits<double>::max();
  size_t no_change = 0;
  while (num_iterations++ < max_iter && no_change++ < no_change_stop) {
    // Assignment step
    for (size_t i = 0; i < num_pins; i++) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = TimingPropagator::calcDist(load_pins[i]->get_location(), centers[j]);
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
    for (size_t i = 0; i < num_pins; i++) {
      int center_index = assignments[i];
      new_centers[center_index] += load_pins[i]->get_location();
      center_counts[center_index]++;
    }
    for (size_t i = 0; i < k; i++) {
      if (center_counts[i] > 0) {
        new_centers[i] /= center_counts[i];
      } else {
        // random choose a sink as new center
        auto loc = load_pins[dis(gen)]->get_location();
        new_centers[i] = loc;
      }
    }
    centers = new_centers;
    // Check cap variance
    std::vector<double> cluster_cap(k, 0);
    for (size_t i = 0; i < num_pins; i++) {
      int center_index = assignments[i];
      cluster_cap[center_index] += load_pins[i]->get_cap_load();
    }
    double cap_variance = calcVariance(cluster_cap);
    // Check for convergence
    if (cap_variance < prev_cap_variance) {
      best_clusters.clear();
      best_clusters.resize(k);
      prev_cap_variance = cap_variance;
      for (size_t i = 0; i < num_pins; i++) {
        int center_index = assignments[i];
        best_clusters[center_index].push_back(load_pins[i]);
      }
      no_change = 0;
    }
  }
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Pin*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
/**
 * @brief normal kmeans
 *
 * @param load_pins
 * @param k
 * @param seed
 * @param max_iter
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::kMeans(const std::vector<Pin*>& load_pins, const size_t& k, const int& seed,
                                                         const size_t& max_iter)
{
  size_t num_pins = load_pins.size();
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_pins - 1);

  std::vector<int> assignments(num_pins);
  std::vector<Point> centers(k);
  for (size_t i = 0; i < k; ++i) {
    auto loc = load_pins[i]->get_location();
    centers[i] = loc;
  }
  for (size_t iter = 0; iter < max_iter; ++iter) {
    std::vector<double> new_center_x(k, 0);
    std::vector<double> new_center_y(k, 0);
    std::vector<Point> new_centers(k);
    std::vector<int> center_counts(k, 0);
    for (size_t i = 0; i < num_pins; ++i) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < k; ++j) {
        double distance = TimingPropagator::calcDist(load_pins[i]->get_location(), centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
      new_center_x[min_center_index] += load_pins[i]->get_location().x();
      new_center_y[min_center_index] += load_pins[i]->get_location().y();
      center_counts[min_center_index]++;
    }
    for (size_t i = 0; i < k; ++i) {
      if (center_counts[i] > 0) {
        new_centers[i] = Point(new_center_x[i] / center_counts[i], new_center_y[i] / center_counts[i]);
      } else {
        // random choose a sink as new center
        auto loc = load_pins[dis(gen)]->get_location();
        new_centers[i] = loc;
      }
    }
    if (new_centers == centers) {
      break;
    }
    centers = new_centers;
  }
  std::vector<std::vector<Pin*>> best_clusters(k);
  for (size_t i = 0; i < num_pins; ++i) {
    int center_index = assignments[i];
    best_clusters[center_index].push_back(load_pins[i]);
  }
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Pin*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
/**
 * @brief iter clustering
 *
 * @param load_pins
 * @param max_fanout
 * @param iters
 * @param no_change_stop
 * @param limit_ratio
 * @param log
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::iterClustering(const std::vector<Pin*>& load_pins, const size_t& max_fanout,
                                                                 const size_t& iters, const size_t& no_change_stop,
                                                                 const double& limit_ratio, const bool& log)
{
  LOG_FATAL_IF(max_fanout < 2) << "max_fanout should be greater than 1";
  if (load_pins.size() == 2) {
    return {load_pins};
  }
  LOG_INFO_IF(log) << "iterative clustering";
  const size_t max_num = 40000;
  if (load_pins.size() > max_num) {
    auto divide_num = 4;
    LOG_INFO << "Inst num: " << load_pins.size() << ", K-Means init clustering to " << divide_num << " clusters";
    auto divide_clusters = kMeansPlus(load_pins, divide_num, 0, iters, no_change_stop);
    std::vector<std::vector<Pin*>> clusters;
    std::ranges::for_each(divide_clusters, [&](const std::vector<Pin*>& divide_cluster) {
      auto sub_cluster = iterClustering(divide_cluster, max_fanout, iters, no_change_stop, limit_ratio, log);
      clusters.insert(clusters.end(), sub_cluster.begin(), sub_cluster.end());
    });
    return clusters;
  }

  // initialize clusters
  size_t cluster_num = std::ceil(1.0 * load_pins.size() / (limit_ratio * max_fanout));
  if (cluster_num == load_pins.size()) {
    cluster_num = load_pins.size() - 1;
  }
  auto clusters = kMeansPlus(load_pins, cluster_num);
  int max_fanout_temp = max_fanout;
  if (cluster_num > clusters.size()) {
    cluster_num = clusters.size();
    max_fanout_temp = std::ceil(1.0 * load_pins.size() / cluster_num);
  }
  size_t kmeans_num
      = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const std::vector<Pin*>& c) { return sum + c.size(); });
  LOG_FATAL_IF(kmeans_num != load_pins.size()) << "num of load_pins is not equal to num of clusters (kmeans load_pins num: " << kmeans_num
                                               << ", load_pins num: " << load_pins.size() << ")";
  LOG_WARNING_IF(max_fanout_temp * clusters.size() < load_pins.size())
      << "multiply max_fanout by num of clusters is less than num of load_pins, "
      << "max_fanout: " << max_fanout_temp << ", clusters num: " << clusters.size() << ", load_pins num: " << load_pins.size();
  LOG_INFO_IF(log) << "initial clusters: " << clusters.size();
  LOG_FATAL_IF(cluster_num != clusters.size()) << "num of cluster num and clusters.size(), cluster_num : " << cluster_num
                                               << ", clusters.size(): " << clusters.size() << ")";
  if (clusters.size() == 1) {
    return clusters;
  }
  // make temp centroid buffer
  std::vector<Point> centers = getCentroids(clusters);
  auto best_clusters = clusters;
  auto kmeans_var = calcBalanceVariance(clusters, centers);
  auto mcf_var = std::numeric_limits<double>::max();
  LOG_INFO_IF(log) << "initial kmeans var: " << kmeans_var;
  size_t no_change = 0;
  // iterate
  for (size_t i = 0; i < iters; ++i, ++no_change) {
    if (no_change > no_change_stop) {
      LOG_INFO_IF(log) << "no change should stop in iter: " << i;
      break;
    }
    if ((i + 1) % 50 == 0) {
      LOG_INFO_IF(log) << "iter: " << i + 1;
    }
    MinCostFlow<Pin*> mcf;
    std::ranges::for_each(load_pins, [&mcf](Pin* pin) { mcf.add_node(pin->get_location().x(), pin->get_location().y(), pin); });
    std::ranges::for_each(centers, [&mcf](const Point& center) { mcf.add_center(center.x(), center.y()); });
    clusters = mcf.run(max_fanout_temp);
    size_t mcf_num
        = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const std::vector<Pin*>& c) { return sum + c.size(); });
    LOG_FATAL_IF(mcf_num != load_pins.size()) << "num of load_pins is not equal to num of min-cost-flow result (mcf load_pins num: "
                                              << mcf_num << ", load_pins num: " << load_pins.size() << ")";
    centers = getCentroids(clusters);
    auto new_mcf_var = calcBalanceVariance(clusters, centers);
    if (new_mcf_var < mcf_var) {
      mcf_var = new_mcf_var;
      best_clusters = clusters;
      no_change = 0;
      LOG_INFO_IF(log) << "update in mcf iter: " << i + 1;
    } else {
      clusters = kMeansPlus(load_pins, cluster_num, i, 5);
      auto temp_centers = getCentroids(clusters);
      auto temp_kmeans_var = calcBalanceVariance(clusters, temp_centers);
      if (temp_kmeans_var < kmeans_var) {
        kmeans_var = temp_kmeans_var;
        centers = temp_centers;
        no_change = 0;
        LOG_INFO_IF(log) << "update in kmeans iter: " << i + 1;
      }
    }
  }
  LOG_INFO_IF(log) << "final mcf var: " << mcf_var;
  LOG_INFO_IF(log) << "final kmeans var: " << kmeans_var;
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Pin*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
/**
 * @brief slack net length clustering
 *
 * @param clusters
 * @param max_net_length
 * @param max_fanout
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::slackClustering(const std::vector<std::vector<Pin*>>& clusters,
                                                                  const double& max_net_length, const size_t& max_fanout)
{
  std::vector<std::vector<Pin*>> slack_clusters;
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) {
    auto est_net_length = estimateNetLength(cluster);
    auto hpwl = calcHPWL(cluster);
    if (hpwl < TimingPropagator::getMinLength() || est_net_length < max_net_length) {
      slack_clusters.push_back(cluster);
    } else {
      // reclustering
      size_t recluster_num = std::ceil(est_net_length / max_net_length);
      recluster_num = std::min(recluster_num, cluster.size());
      if (recluster_num == 1) {
        slack_clusters.push_back(cluster);
        return;
      }
      auto new_clusters = kMeansPlus(cluster, recluster_num);
      std::ranges::for_each(new_clusters, [&](const std::vector<Pin*>& new_cluster) {
        auto re_slack_clusters = slackClustering({new_cluster}, max_net_length, max_fanout);
        slack_clusters.insert(slack_clusters.end(), re_slack_clusters.begin(), re_slack_clusters.end());
      });
    }
  });
  return slack_clusters;
}
/**
 * @brief clustering enhancement by AnnealOpt
 *
 * @param clusters
 * @param max_fanout
 * @param max_cap
 * @param max_net_length
 * @param skew_bound
 * @param max_iter
 * @param cooling_rate
 * @param temperature
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::clusteringEnhancement(const std::vector<std::vector<Pin*>>& clusters,
                                                                        const int& max_fanout, const double& max_cap,
                                                                        const double& max_net_length, const double& skew_bound,
                                                                        const size_t& max_iter, const double& cooling_rate,
                                                                        const double& temperature)
{
  if (clusters.size() == 1) {
    return clusters;
  }
  VioAnnealOpt solver(clusters);
  solver.initParameter(max_iter, cooling_rate, temperature, max_fanout, max_cap, max_net_length, skew_bound);
  solver.automaticTemperature();
  bool log = false;
#ifdef DEBUG_ICTS_ANNEAL_OPT
  log = true;
#endif
  return solver.run(log);
}
/**
 * @brief get all cluster guide center
 *
 * @param clusters
 * @param center
 * @param level
 * @return std::vector<Point>
 */
std::vector<Point> BalanceClustering::guideCenter(const std::vector<std::vector<Pin*>>& clusters, const std::optional<Point>& center,
                                                  const double& min_length, const size_t& level)
{
  std::vector<Pin*> load_pins;
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) {
    std::transform(cluster.begin(), cluster.end(), std::back_inserter(load_pins), [](Pin* pin) { return pin; });
  });
  auto* buf = TreeBuilder::defaultTree("temp", load_pins, std::nullopt, center.value_or(calcCentroid(load_pins)), TopoType::kBiPartition);
  // auto* buf = TreeBuilder::genBufInst("temp", center.value_or(calcCentroid(load_pins)));
  auto* driver_pin = buf->get_driver_pin();
  // TreeBuilder::shallowLightTree("temp", driver_pin, pins);

  // find communis ancestor
  LCA solver(driver_pin);
  std::vector<Point> centers;
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) {
    Point guide_loc = Point(-1, -1);
    if (cluster.size() == 1) {
      auto* load = cluster.front();
      auto* parent = load->get_parent();
      while (TimingPropagator::calcLen(parent->get_location(), load->get_location()) < min_length && parent->get_parent()) {
        parent = parent->get_parent();
      }
      guide_loc = parent->get_location();
    } else {
      auto center = calcCentroid(cluster);
      std::vector<Node*> nodes;
      std::ranges::for_each(cluster, [&nodes](Pin* pin) { nodes.push_back(pin); });
      auto* lca = solver.query(nodes);
      size_t lca_level = 1;
      while (lca_level < level && lca->get_parent()) {
        lca = lca->get_parent();
        ++lca_level;
      }
      while (TimingPropagator::calcLen(lca->get_location(), center) < min_length && lca->get_parent()) {
        lca = lca->get_parent();
      }
      guide_loc = lca->get_location();
    }
    centers.push_back(guide_loc);
  });
  auto* net = TimingPropagator::genNet("temp", driver_pin, load_pins);
  TimingPropagator::resetNet(net);
  return centers;
}
/**
 * @brief get the min delay cluster
 *
 * @param clusters
 * @return std::vector<Pin*>
 */
std::vector<Pin*> BalanceClustering::getMinDelayCluster(const std::vector<std::vector<Pin*>>& clusters)
{
  std::map<size_t, double> delay_map;
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    delay_map[i] = estimateNetDelay(cluster, false);
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
 * @return std::vector<Pin*>
 */
std::vector<Pin*> BalanceClustering::getMaxDelayCluster(const std::vector<std::vector<Pin*>>& clusters)
{
  std::map<size_t, double> delay_map;
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    delay_map[i] = estimateNetDelay(cluster);
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
 * @return std::vector<Pin*>
 */
std::vector<Pin*> BalanceClustering::getWorstViolationCluster(const std::vector<std::vector<Pin*>>& clusters)
{
  // sort violation by skew, then cap, then net length
  auto vio_cmp = [](const ViolationScore& a, const ViolationScore& b) {
    return a.skew_vio_score > b.skew_vio_score || (a.skew_vio_score == b.skew_vio_score && a.cap_vio_score > b.cap_vio_score)
           || (a.skew_vio_score == b.skew_vio_score && a.cap_vio_score == b.cap_vio_score && a.net_len_vio_score > b.net_len_vio_score);
  };
  std::vector<ViolationScore> vio_scores;
  std::ranges::for_each(clusters, [&vio_scores](const std::vector<Pin*>& cluster) {
    auto score = calcScore(cluster);
    vio_scores.push_back(score);
  });
  std::ranges::sort(vio_scores, vio_cmp);
  auto is_violation = [&](const ViolationScore& vio_score) {
    if (vio_score.skew_vio_score > TimingPropagator::getSkewBound() || vio_score.cap_vio_score > TimingPropagator::getMaxCap()
        || vio_score.net_len_vio_score > TimingPropagator::getMaxLength()) {
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
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> BalanceClustering::getMostRecentClusters(const std::vector<std::vector<Pin*>>& clusters,
                                                                        const std::vector<Pin*>& center_cluster, const size_t& num_limit,
                                                                        const size_t& cluster_num_limit)
{
  std::vector<std::pair<size_t, double>> id_dist_pairs;
  auto loc = calcCentroid(center_cluster);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto center = calcCentroid(clusters[i]);
    auto dist = TimingPropagator::calcDist(loc, center);
    id_dist_pairs.push_back({i, dist});
  }
  std::ranges::sort(id_dist_pairs,
                    [](const std::pair<size_t, double>& p1, const std::pair<size_t, double>& p2) { return p1.second < p2.second; });
  std::vector<std::vector<Pin*>> recent_clusters;
  size_t pin_num = center_cluster.size();
  for (auto id_dist_pair : id_dist_pairs) {
    auto id = id_dist_pair.first;
    auto cluster = clusters[id];
    pin_num += cluster.size();
    if (pin_num > num_limit && !recent_clusters.empty()) {
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
std::vector<Point> BalanceClustering::getCentroids(const std::vector<std::vector<Pin*>>& clusters)
{
  std::vector<Point> buffers;
  std::ranges::for_each(clusters, [&buffers](const std::vector<Pin*>& cluster) {
    auto centroid = calcCentroid(cluster);
    buffers.push_back(centroid);
  });
  return buffers;
}
/**
 * @brief divide load_pins by function to pair<lower: num * ratio, higher: num * (1 - ratio)>
 *
 * @param load_pins
 * @return std::pair<std::vector<Pin*>, std::vector<Pin*>>
 */
std::pair<std::vector<Pin*>, std::vector<Pin*>> BalanceClustering::divideBy(const std::vector<Pin*>& load_pins,
                                                                            const std::function<double(Pin*)>& func, const double& ratio)
{
  auto cmp = [&](Pin* pin_1, Pin* pin_2) { return func(pin_1) < func(pin_2); };
  std::vector<Pin*> sorted_load_pins = load_pins;
  std::ranges::sort(sorted_load_pins, cmp);
  size_t num = sorted_load_pins.size();
  size_t split_id = num * (1 - ratio);
  std::vector<Pin*> left(sorted_load_pins.begin(), sorted_load_pins.begin() + split_id);
  std::vector<Pin*> right(sorted_load_pins.begin() + split_id, sorted_load_pins.end());
  return {left, right};
}
/**
 * @brief get the clusters on the boundary
 *
 * @param clusters
 * @return std::pair<std::vector<std::vector<Pin*>>, std::vector<std::vector<Pin*>>>
 */
std::pair<std::vector<std::vector<Pin*>>, std::vector<std::vector<Pin*>>> BalanceClustering::getBoundCluster(
    const std::vector<std::vector<Pin*>>& clusters)
{
  auto centers = getCentroids(clusters);

  auto convex = centers;
  convexHull(convex);
  // Filter all clusters where center is on the convex package
  auto remain_centers = centers;
  auto is_in_convex = [&convex](const Point& center) { return std::find(convex.begin(), convex.end(), center) != convex.end(); };
  remain_centers.erase(
      std::remove_if(remain_centers.begin(), remain_centers.end(), [&](const Point& center) { return is_in_convex(center); }),
      remain_centers.end());
  auto second_convex = remain_centers;
  if (!second_convex.empty()) {
    convexHull(second_convex);
  }
  convex.insert(convex.end(), second_convex.begin(), second_convex.end());

  std::vector<std::vector<Pin*>> bound_clusters;
  std::vector<std::vector<Pin*>> remain_clusters;
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) {
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
void BalanceClustering::latencyOpt(const std::vector<Pin*>& cluster, const double& skew_bound, const double& ratio)
{
  if (cluster.size() * ratio < 1 || cluster.size() * (1 - ratio) < 1) {
    return;
  }
  auto get_max_delay = [&](Pin* pin) {
    auto* inst = pin->get_inst();
    if (inst->isSink()) {
      return 0.0;
    }
    TimingPropagator::initLoadPinDelay(pin, true);
    return pin->get_max_delay();
  };
  auto divide = divideBy(cluster, get_max_delay, ratio);
  auto feasible_load_pins = divide.first;
  auto delay_bound = feasible_load_pins.back()->get_max_delay();
  auto to_be_amplify = divide.second;
  std::ranges::for_each(to_be_amplify, [&](Pin* pin) {
    auto* inst = pin->get_inst();
    auto* driver_pin = inst->get_driver_pin();
    auto* net = driver_pin->get_net();
    auto feasible_cell = TreeBuilder::feasibleCell(inst, skew_bound);
    for (auto cell : feasible_cell) {
      inst->set_cell_master(cell);
      TimingPropagator::update(net);
      TimingPropagator::initLoadPinDelay(pin, true);
      auto max_delay = pin->get_max_delay();
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
double BalanceClustering::estimateSkew(const std::vector<Pin*>& cluster)
{
  auto center = calcCentroid(cluster);
  auto buffer = TreeBuilder::genBufInst("temp", center);
  // set cell master
  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  // location legitimization
  auto* driver_pin = buffer->get_driver_pin();
  TreeBuilder::localPlace(driver_pin, cluster);
  if (cluster.size() == 1) {
    auto load_pin = cluster.front();
    TreeBuilder::directConnectTree(driver_pin, load_pin);
  } else {
    TreeBuilder::shallowLightTree("Salt", driver_pin, cluster);
  }
  auto* net = TimingPropagator::genNet("temp", driver_pin, cluster);
  TimingPropagator::update(net);
  auto skew = driver_pin->get_max_delay() - driver_pin->get_min_delay();

  TimingPropagator::resetNet(net);

  return skew;
}
/**
 * @brief estimate net delay of cluster
 *
 * @param cluster
 * @param is_max
 * @return double
 */
double BalanceClustering::estimateNetDelay(const std::vector<Pin*>& cluster, const bool& is_max)
{
  auto net_len = estimateNetLength(cluster);
  auto total_cap = estimateNetCap(cluster);
  double res = net_len * TimingPropagator::getUnitRes();
  auto net_delay = total_cap * res / 2;
  auto target_delay = is_max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
  std::ranges::for_each(cluster, [&net_delay, &target_delay, &is_max](const Pin* pin) {
    auto* inst = pin->get_inst();
    auto cur_delay = (inst->isBuffer() ? inst->get_driver_pin()->get_max_delay() : 0) + net_delay;
    target_delay = is_max ? std::max(target_delay, cur_delay) : std::min(target_delay, cur_delay);
  });
  return target_delay;
}
/**
 * @brief  estimate cap of cluster
 *
 * @param cluster
 * @return double
 */
double BalanceClustering::estimateNetCap(const std::vector<Pin*>& cluster)
{
  auto net_len = estimateNetLength(cluster);
  auto pin_cap
      = std::accumulate(cluster.begin(), cluster.end(), 0.0, [](double total, const Pin* pin) { return total + pin->get_cap_load(); });
  auto total_cap = net_len * TimingPropagator::getUnitCap() + pin_cap;
  return total_cap;
}
/**
 * @brief estimate net length of cluster
 *
 * @param cluster
 * @return double
 */
double BalanceClustering::estimateNetLength(const std::vector<Pin*>& cluster)
{
  if (cluster.size() <= 1) {
    return 0;
  }
  // auto center = calcCentroid(cluster);
  auto load_pins = cluster;
  TreeBuilder::localPlace(load_pins);
  auto* buf = TreeBuilder::defaultTree("temp", cluster, TimingPropagator::getSkewBound(), std::nullopt, TopoType::kBiPartition);
  auto* driver_pin = buf->get_driver_pin();
  TreeBuilder::localPlace(driver_pin, cluster);
  auto* net = TimingPropagator::genNet("temp", driver_pin, cluster);
  TimingPropagator::update(net);
  auto net_len = driver_pin->get_sub_len();
  TimingPropagator::resetNet(net);
  return net_len;
}
/**
 * @brief calculate centroid of cluster
 *
 * @param cluster
 * @return Point
 */
Point BalanceClustering::calcCentroid(const std::vector<Pin*>& cluster)
{
  int64_t x = 0;
  int64_t y = 0;
  x = std::accumulate(cluster.begin(), cluster.end(), x, [](int64_t total, const Pin* pin) { return total + pin->get_location().x(); });
  y = std::accumulate(cluster.begin(), cluster.end(), y, [](int64_t total, const Pin* pin) { return total + pin->get_location().y(); });

  return Point(x / cluster.size(), y / cluster.size());
}
/**
 * @brief calculate bound centroid of cluster
 *
 * @param cluster
 * @return Point
 */
Point BalanceClustering::calcBoundCentroid(const std::vector<Pin*>& cluster)
{
  auto min_x = std::numeric_limits<int>::max();
  auto min_y = std::numeric_limits<int>::max();
  auto max_x = std::numeric_limits<int>::min();
  auto max_y = std::numeric_limits<int>::min();
  std::ranges::for_each(cluster, [&min_x, &min_y, &max_x, &max_y](const Pin* pin) {
    auto loc = pin->get_location();
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
int BalanceClustering::calcHPMD(const std::vector<Pin*>& cluster)
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  std::ranges::for_each(cluster, [&min_x, &min_y, &max_x, &max_y](const Pin* pin) {
    auto loc = pin->get_location();
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
double BalanceClustering::calcHPWL(const std::vector<Pin*>& cluster)
{
  auto hpmd = calcHPMD(cluster);
  return 1.0 * hpmd / TimingPropagator::getDbUnit();
}
/**
 * @brief calculate sum of absolute error
 *
 * @param clusters
 * @param centers
 * @return double
 */
double BalanceClustering::calcSAE(const std::vector<std::vector<Pin*>>& clusters, const std::vector<Point>& centers)
{
  double sae = 0;
  for (size_t i = 0; i < clusters.size(); ++i) {
    sae += std::accumulate(clusters[i].begin(), clusters[i].end(), 0.0, [&centers, &i](double total, const Pin* pin) {
      return total + TimingPropagator::calcDist(pin->get_location(), centers[i]);
    });
  }
  return sae;
}
/**
 * @brief trade off between cap and delay
 *
 * @param clusters
 * @param centers
 * @param cap_coef
 * @param delay_coef
 * @return double
 */
double BalanceClustering::calcBalanceVariance(const std::vector<std::vector<Pin*>>& clusters, const std::vector<Point>& centers,
                                              const double& cap_coef, const double& delay_coef)
{
  return cap_coef * calcCapVariance(clusters, centers) + delay_coef * calcDelayVariance(clusters, centers);
}
/**
 * @brief calculate cap variance
 *
 * @param clusters
 * @param centers
 * @return double
 */
double BalanceClustering::calcCapVariance(const std::vector<std::vector<Pin*>>& clusters, const std::vector<Point>& centers)
{
  // cap variance
  std::vector<double> cluster_cap(clusters.size(), 0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    cluster_cap[i] = estimateNetCap(cluster);
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
 * @param centers
 * @return double
 */
double BalanceClustering::calcDelayVariance(const std::vector<std::vector<Pin*>>& clusters, const std::vector<Point>& centers)
{
  // delay variance
  std::vector<double> cluster_delay(clusters.size(), 0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    cluster_delay[i] = estimateNetDelay(cluster);
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
 * @return ViolationScore
 */
ViolationScore BalanceClustering::calcScore(const std::vector<Pin*>& cluster)
{
  auto skew_vio_score = estimateSkew(cluster);
  auto cap_vio_score = estimateNetCap(cluster);
  auto net_len_vio_score = estimateNetLength(cluster);
  return ViolationScore{skew_vio_score, cap_vio_score, net_len_vio_score, cluster};
}
/**
 * @brief calculate cross product of three points
 *
 * @param p1
 * @param p2
 * @param p3
 * @return double
 */
double BalanceClustering::crossProduct(const Point& p1, const Point& p2, const Point& p3)
{
  return 1.0 * (p2.x() - p1.x()) * (p3.y() - p1.y()) - 1.0 * (p3.x() - p1.x()) * (p2.y() - p1.y());
}
/**
 * @brief calculate convex hull of points
 *
 * @param pts
 */
void BalanceClustering::convexHull(std::vector<Point>& pts)
{
  // check pts num
  if (pts.size() < 2) {
    return;
  }
  if (pts.size() == 2) {
    auto dist = TimingPropagator::calcDist(pts.front(), pts.back());
    if (dist == 0) {
      pts.pop_back();
    }
    return;
  }
  // calculate convex hull by Andrew algorithm
  std::ranges::sort(pts);
  std::vector<Point> ans(2 * pts.size());
  size_t k = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    while (k > 1 && crossProduct(ans[k - 2], ans[k - 1], pts[i]) <= 0) {
      --k;
    }
    ans[k++] = pts[i];
  }
  for (size_t i = pts.size() - 1, t = k + 1; i > 0; --i) {
    while (k >= t && crossProduct(ans[k - 2], ans[k - 1], pts[i - 1]) <= 0) {
      --k;
    }
    ans[k++] = pts[i - 1];
  }
  pts = std::vector<Point>(ans.begin(), ans.begin() + k - 1);
}
/**
 * @brief find the pareto front of points
 *
 * @param pts
 * @return std::vector<CtsPoint<double>>
 */
std::vector<CtsPoint<double>> BalanceClustering::paretoFront(const std::vector<CtsPoint<double>>& pts)
{
  auto pareto_pts = pts;
  auto cmp = [](const CtsPoint<double>& p1, const CtsPoint<double>& p2) {
    return p1.x() < p2.x() || (std::abs(p1.x() - p2.x()) < std::numeric_limits<double>::epsilon() && p1.y() < p2.y());
  };
  std::ranges::sort(pareto_pts, cmp);
  std::vector<CtsPoint<double>> ans;
  double min_y = std::numeric_limits<double>::max();
  for (size_t i = 0; i < pareto_pts.size(); ++i) {
    auto p = pareto_pts[i];
    if (i == 0 || p.y() < min_y) {
      ans.push_back(p);
      min_y = p.y();
    }
  }
  return ans;
}
/**
 * @brief check whether p is in pts
 *
 * @param p
 * @param pts
 * @return true
 * @return false
 */
bool BalanceClustering::isContain(const Point& p, const std::vector<Point>& pts)
{
  auto on_line = [](const Point& p, const std::pair<Point, Point>& line) {
    auto s_x = line.first.x();
    auto s_y = line.first.y();
    auto t_x = line.second.x();
    auto t_y = line.second.y();
    auto p_x = p.x();
    auto p_y = p.y();
    auto cross_product = (p_x - s_x) * (t_y - s_y) - (p_y - s_y) * (t_x - s_x);
    if (cross_product != 0) {
      return false;
    }
    if (std::min(s_x, t_x) <= p_x && p_x <= std::max(s_x, t_x) && std::min(s_y, t_y) <= p_y && p_y <= std::max(s_y, t_y)) {
      return true;
    }
    return false;
  };

  auto bound = pts;
  convexHull(bound);
  auto is_in_bound = false;
  auto p_x = p.x();
  auto p_y = p.y();
  auto n = bound.size();
  auto j = n - 1;
  for (size_t i = 0; i < n; j = i, ++i) {
    auto s_x = bound[i].x();
    auto s_y = bound[i].y();
    auto t_x = bound[j].x();
    auto t_y = bound[j].y();
    if ((s_y < p_y && t_y >= p_y) || (t_y < p_y && s_y >= p_y)) {
      if (s_x + (p_y - s_y) / (t_y - s_y) * (t_x - s_x) < p_x) {
        is_in_bound = !is_in_bound;
      }
    }
  }
  if (is_in_bound) {
    return true;
  }
  auto pt = p;
  for (size_t i = 0; i < bound.size(); ++i) {
    auto j = (i + 1) % bound.size();
    if (on_line(pt, {bound[i], bound[j]})) {
      return true;
    }
  }
  return false;
}
/**
 * @brief judge whether two clusters are the same
 *
 * @param clusters1
 * @param clusters2
 * @return true
 * @return false
 */
bool BalanceClustering::isSame(const std::vector<std::vector<Pin*>>& clusters1, const std::vector<std::vector<Pin*>>& clusters2)
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
void BalanceClustering::writeClusterPy(const std::vector<std::vector<Pin*>>& clusters, const std::string& save_name)
{
  LOG_INFO << "Writing clusters to python file...";
  // write the cluster to python file
  auto* config = CTSAPIInst.get_config();
  auto path = config->get_work_dir() + "/file";
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  }
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
    // for (size_t j = 0; j < clusters[i].size(); ++j) {
    //   ofs << "plt.text(" << clusters[i][j]->get_location().x() << ", " << clusters[i][j]->get_location().y() << ", '"
    //       << clusters[i][j]->getCapLoad() << "', fontsize=3)" << std::endl;
    // }
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