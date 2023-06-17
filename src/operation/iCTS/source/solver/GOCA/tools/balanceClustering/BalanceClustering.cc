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

#include <limits>
#include <random>
#include <ranges>

#include "CTSAPI.hpp"
#include "log/Log.hh"
#include "minCostFlow/MinCostFlow.hh"
namespace icts {
/**
 * @brief init clustering
 *
 * @param sinks
 * @param k
 * @param seed
 * @param max_iter
 * @return std::vector<std::vector<Node*>>
 */
std::vector<std::vector<Node*>> BalanceClustering::kMeans(const std::vector<Node*>& sinks, const size_t& k, const int& seed,
                                                          const size_t& max_iter)
{
  std::vector<std::vector<Node*>> best_clusters(k);

  std::vector<Point> centers;
  size_t num_instances = sinks.size();
  std::vector<int> assignments(num_instances);

  // Randomly choose first center from instances
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_instances - 1);
  auto loc = sinks[dis(gen)]->get_location();
  centers.emplace_back(Point(loc.x(), loc.y()));
  // Choose k-1 remaining centers using kmeans++ algorithm
  while (centers.size() < k) {
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(sinks[i]->get_location(), centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }

    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    auto select_loc = sinks[selected_index]->get_location();
    centers.emplace_back(Point(select_loc.x(), select_loc.y()));
  }

  size_t num_iterations = 0;
  double prev_cap_variance = std::numeric_limits<double>::max();
  while (num_iterations++ < max_iter) {
    // Assignment step
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(sinks[i]->get_location(), centers[j]);
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
      new_centers[center_index] += sinks[i]->get_location();
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
      cluster_cap[center_index] += sinks[i]->get_cap_load();
    }
    double sum = std::accumulate(std::begin(cluster_cap), std::end(cluster_cap), 0.0);
    double mean = sum / cluster_cap.size();
    double cap_variance = 0.0;
    std::ranges::for_each(cluster_cap, [&](const double d) { cap_variance += std::pow(d - mean, 2); });
    cap_variance /= cluster_cap.size();
    // Check for convergence
    if (cap_variance < prev_cap_variance) {
      best_clusters.clear();
      best_clusters.resize(k);
      prev_cap_variance = cap_variance;
      for (size_t i = 0; i < num_instances; i++) {
        int center_index = assignments[i];
        best_clusters[center_index].push_back(sinks[i]);
      }
    }
  }
  return best_clusters;
}
/**
 * @brief iter clustering
 *
 * @param sinks
 * @param max_dist
 * @param max_fanout
 * @param iters
 * @param no_change_stop
 * @param limit_ratio
 * @return std::vector<std::vector<Node*>>
 */
std::vector<std::vector<Node*>> BalanceClustering::iterClustering(const std::vector<Node*>& sinks, const double& max_dist,
                                                                  const size_t& max_fanout, const size_t& iters,
                                                                  const size_t& no_change_stop, const double& limit_ratio)
{
  LOG_INFO << "iterative clustering";
  // initialize clusters
  size_t cluster_num = std::ceil(1.0 * sinks.size() / (limit_ratio * max_fanout));
  auto clusters = kMeans(sinks, cluster_num);
  LOG_FATAL_IF(max_fanout * clusters.size() < sinks.size()) << "multiply max_fanout by num of clusters is less than num of sinks";
  LOG_INFO << "initial clusters: " << clusters.size();
  // make temp centroid buffer
  std::vector<Point> buffers = genCentroidBuffers(clusters);
  auto best_clusters = clusters;
  auto cap_var = calcCapVariance(clusters, buffers);
  LOG_INFO << "initial cap var: " << cap_var;
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
    MinCostFlow<Node*> mcf;
    std::ranges::for_each(sinks, [&mcf](Node* sink) { mcf.add_node(sink->get_location().x(), sink->get_location().y(), sink); });
    std::ranges::for_each(buffers, [&mcf](const Point& buffer) { mcf.add_center(buffer.x(), buffer.y()); });
    clusters = mcf.run(max_dist, max_fanout);
    buffers = genCentroidBuffers(clusters);
    auto new_cap_var = calcCapVariance(clusters, buffers);
    if (new_cap_var < cap_var) {
      cap_var = new_cap_var;
      best_clusters = clusters;
      LOG_INFO << "update in iter: " << i + 1;
      no_change = 0;
    } else {
      clusters = kMeans(sinks, cluster_num, i);
      auto temp_buffers = genCentroidBuffers(clusters);
      auto temp_new_cap_var = calcCapVariance(clusters, temp_buffers);
      if (temp_new_cap_var < cap_var) {
        cap_var = temp_new_cap_var;
        buffers = temp_buffers;
        best_clusters = clusters;
        no_change = 0;
        LOG_INFO << "update in kmeans iter: " << i + 1;
      }
    }
  }
  LOG_INFO << "final cap var: " << cap_var;
  return best_clusters;
}
/**
 * @brief generate centroid buffers location
 * 
 * @param clusters 
 * @return std::vector<Point> 
 */
std::vector<Point> BalanceClustering::genCentroidBuffers(const std::vector<std::vector<Node*>>& clusters)
{
  std::vector<Point> buffers;
  for (const auto& cluster : clusters) {
    double sum_x = 0, sum_y = 0, sum_cap = 0;
    for (const auto& sink : cluster) {
      auto loc = sink->get_location();
      sum_x += loc.x();
      sum_y += loc.y();
      sum_cap += sink->get_cap_load();
    }
    buffers.emplace_back(Point(sum_x / cluster.size(), sum_y / cluster.size()));
  }
  return buffers;
}
/**
 * @brief calculate sum of absolute error
 * 
 * @param clusters 
 * @param buffers 
 * @return double 
 */
double BalanceClustering::calcSAE(const std::vector<std::vector<Node*>>& clusters, const std::vector<Point>& buffers)
{
  double sae = 0;
  for (size_t i = 0; i < clusters.size(); ++i) {
    for (const auto& sink : clusters[i]) {
      sae += pgl::manhattan_distance(sink->get_location(), buffers[i]);
    }
  }
  return sae;
}
/**
 * @brief calculate cap variance
 * 
 * @param clusters 
 * @param buffers 
 * @return double 
 */
double BalanceClustering::calcCapVariance(const std::vector<std::vector<Node*>>& clusters, const std::vector<Point>& buffers)
{
  // Check cap variance
  std::vector<double> cluster_cap(clusters.size(), 0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    int x_min = std::numeric_limits<int>::max(), x_max = std::numeric_limits<int>::min(), y_min = std::numeric_limits<int>::max(),
        y_max = std::numeric_limits<int>::min();
    for (auto sink : cluster) {
      auto loc = sink->get_location();
      x_min = std::min(x_min, loc.x());
      x_max = std::max(x_max, loc.x());
      y_min = std::min(y_min, loc.y());
      y_max = std::max(y_max, loc.y());
      cluster_cap[i] += sink->get_cap_load();
    }
    auto hp_wl = (x_max - x_min) + (y_max - y_min);
    auto stn_wl = 2.5 * hp_wl;
    cluster_cap[i] += stn_wl * CTSAPIInst.getClockUnitCap();
  }
  double sum = std::accumulate(std::begin(cluster_cap), std::end(cluster_cap), 0);
  double mean = sum / cluster_cap.size();
  double cap_variance = 0;
  std::ranges::for_each(cluster_cap, [&cap_variance, &mean](const double d) { cap_variance += std::pow(d - mean, 2); });
  cap_variance /= cluster_cap.size();
  return cap_variance;
}

}  // namespace icts