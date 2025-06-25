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
 * @file AnnealOpt.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "AnnealOpt.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <unordered_set>

#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "log/Log.hh"
namespace icts {
/**
 * @brief automatic temperature
 *
 */
void AnnealOptInterface::automaticTemperature()
{
  _auto_temp = true;
}
/**
 * @brief interface to run the AnnealOpt solver
 *
 * @param log
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> AnnealOptInterface::run(const bool& log)
{
  initCostMap();
  if (_auto_temp) {
    _temperature = _cur_cost / std::ceil(1.0 * _cur_solution.size() / 20);  // TBD optimize
  }
  LOG_INFO_IF(log) << "Start AnnealOpt --- ";
  LOG_INFO_IF(log) << "  Iteration: " << _max_iter;
  LOG_INFO_IF(log) << "  Temperature: " << _temperature;
  LOG_INFO_IF(log) << "  Cooling Rate: " << _cooling_rate;

  auto init_cost = _cur_cost;
  LOG_INFO_IF(log) << "  Initial Cost: " << init_cost;
  _best_cost = _cur_cost;
  if (_best_cost <= std::numeric_limits<float>::epsilon()) {
    LOG_INFO_IF(log) << "  No Need to Optimize!";
    LOG_INFO_IF(log) << "End AnnealOpt --- ";
    return _cur_solution;
  }
  auto temperature = _temperature;
  auto cooling_rate = _cooling_rate;
  auto best_solution = _cur_solution;
  size_t iter = 0;
  while (iter < _max_iter) {
    bool best_update = false;
    auto operation = randomMove(_cur_solution);
    auto new_solution = commitOperation(operation);
    auto delta_cost = _new_cost - _cur_cost;
    if (delta_cost < 0) {
      updateSolution(new_solution, operation);
      if (_new_cost < _best_cost) {
        best_solution = new_solution;
        _best_cost = _new_cost;
        best_update = true;
        _no_change = 0;
      }
    } else {
      auto prob = std::exp(-delta_cost / temperature);
      if (prob > 0.5) {
        updateSolution(new_solution, operation);
      }
    }
    temperature *= cooling_rate;
    ++iter;
    ++_no_change;
    // align the log
    auto update_label = best_update ? " *" : "";
    LOG_INFO_IF(log) << "  Iteration: " << std::left << std::setfill(' ') << std::setw(std::log10(_max_iter) + 1) << iter
                     << "  Temperature: " << std::left << std::setfill(' ') << std::setw(std::log10(_temperature) + 2) << std::fixed
                     << std::setprecision(2) << temperature << "  Cost: " << std::left << std::setfill(' ')
                     << std::setw(std::log10(init_cost) + 3) << std::fixed << std::setprecision(3) << _cur_cost
                     << "  Delta Cost: " << std::left << std::setfill(' ') << std::setw(6) << std::fixed << std::setprecision(3)
                     << delta_cost << "  Best Cost: " << std::left << std::setfill(' ') << std::setw(std::log10(init_cost) + 3)
                     << std::fixed << std::setprecision(3) << _best_cost << update_label;
    if (_best_cost <= std::numeric_limits<float>::epsilon()) {
      LOG_INFO_IF(log) << "  Found Optimal Solution!";
      break;
    }
  }
  LOG_INFO_IF(log) << "End AnnealOpt --- ";
  _improve = (init_cost - _best_cost) / init_cost;
  if (_improve > 0) {
    LOG_INFO << "  Best Cost: " << std::fixed << std::setprecision(3) << _best_cost << "  Improvement: " << _improve * 100 << "%";
    CTSAPIInst.saveToLog("Best Cost: ", _best_cost, "  Improvement: ", _improve * 100, "%");
  } else {
    LOG_WARNING << "  No Improvement!";
  }
  // remove empty
  best_solution.erase(
      std::remove_if(best_solution.begin(), best_solution.end(), [](const std::vector<Pin*>& cluster) { return cluster.empty(); }),
      best_solution.end());
  return best_solution;
}
/**
 * @brief update the solution and cost map
 *
 * @param new_solution
 * @param op
 */
void AnnealOptInterface::updateSolution(const std::vector<std::vector<Pin*>>& new_solution, const Operation& op)
{
  _cur_solution = new_solution;
  _cur_cost = _new_cost;
  updateCostMap(op);
}
/**
 * @brief commit the operation
 *
 * @param op
 * @return std::vector<std::vector<Pin*>>
 */
std::vector<std::vector<Pin*>> AnnealOptInterface::commitOperation(Operation& op)
{
  auto cluster_id = op.cluster_id;
  auto neighbor_id = op.neighbor_id;
  auto pin_id = op.pin_id;

  auto pre_both_cost = _cost_map[cluster_id] + _cost_map[neighbor_id];

  auto new_solution = _cur_solution;
  auto* inst = new_solution[cluster_id][pin_id];
  new_solution[cluster_id].erase(new_solution[cluster_id].begin() + pin_id);
  new_solution[neighbor_id].push_back(inst);
  auto* from_net = buildNet(new_solution[cluster_id]);
  auto* to_net = buildNet(new_solution[neighbor_id]);
  auto from_cost = cost(from_net);
  auto to_cost = cost(to_net);

  TimingPropagator::resetNet(from_net);
  TimingPropagator::resetNet(to_net);

  auto cur_both_cost = from_cost + to_cost;

  _new_cost = _cur_cost - pre_both_cost + cur_both_cost;
  if (_new_cost < TimingPropagator::kEpsilon) {
    _new_cost = 0;
  }
  op.from_cost = from_cost;
  op.to_cost = to_cost;
  return new_solution;
}
/**
 * @brief random move the inst to a new cluster
 *
 * @return Operation
 */
Operation AnnealOptInterface::randomMove(const std::vector<std::vector<Pin*>>& clusters)
{
  // random choose a cluster (not empty)
  auto cluster_id = randomChooseCluster(clusters, 0.03);

  // random choose a inst in the cluster octagon boundary
  auto pin_id = randomChoosePin(clusters[cluster_id]);

  // choose a neighbor cluster (closest cluster or empty cluster)
  auto new_cluster_id = randomChooseNeighbor(clusters, cluster_id, pin_id);

  return Operation{cluster_id, new_cluster_id, pin_id};
}
/**
 * @brief random choose a cluster in the clusters which cost is in the first [ratio * {clusters.size() - empty_num}] clusters
 *
 * @param clusters
 * @param ratio
 * @return size_t
 */
size_t AnnealOptInterface::randomChooseCluster(const std::vector<std::vector<Pin*>>& clusters, const double& ratio)
{
  size_t empty_num = std::ranges::count_if(clusters, [&](const std::vector<Pin*>& cluster) { return cluster.empty(); });
  auto num = static_cast<size_t>(std::ceil((_cur_solution.size() - empty_num) * ratio));
  std::vector<size_t> cluster_id;
  // choose the first num id from _sorted_cluster_id_map (only "num" clusters)
  for (auto it = _sorted_cluster_id_map.begin(); it != _sorted_cluster_id_map.end() && cluster_id.size() < num; ++it) {
    cluster_id.push_back(it->second);
  }
  std::uniform_int_distribution<size_t> cluster_dist(0, cluster_id.size() - 1);
  auto cluster_id_index = cluster_dist(_gen);
  LOG_FATAL_IF(clusters[cluster_id[cluster_id_index]].empty()) << "Empty cluster";
  return cluster_id[cluster_id_index];
}
/**
 * @brief random choose a inst's id in the cluster which is in the boundary
 *
 * @param cluster
 * @return size_t
 */
size_t AnnealOptInterface::randomChoosePin(const std::vector<Pin*>& cluster)
{
  auto bound_id = findBoundId(cluster);
  std::uniform_int_distribution<size_t> inst_dist(0, bound_id.size() - 1);
  auto pin_id = bound_id[inst_dist(_gen)];
  return pin_id;
}
/**
 * @brief random choose a neighbor cluster id
 *
 * @param clusters
 * @param cluster_id
 * @param pin_id
 * @return size_t
 */
size_t AnnealOptInterface::randomChooseNeighbor(const std::vector<std::vector<Pin*>>& clusters, const size_t& cluster_id,
                                                const size_t& pin_id)
{
  auto* inst = clusters[cluster_id][pin_id];

  auto cluster_dist = [&](const Point& pt, const std::vector<Pin*>& cluster) {
    std::vector<Point> bound;
    std::ranges::transform(cluster, std::back_inserter(bound), [](const Pin* inst) { return inst->get_location(); });
    BalanceClustering::convexHull(bound);
    if (BalanceClustering::isContain(pt, bound)) {
      return 0;
    }
    auto min_dist = std::numeric_limits<int>::max();
    std::ranges::for_each(bound, [&](const Point& bound_pt) { min_dist = std::min(min_dist, Point::manhattanDistance(pt, bound_pt)); });
    return min_dist;
  };

  std::vector<double> dist;

  std::ranges::transform(clusters, std::back_inserter(dist), [&](const std::vector<Pin*>& cluster) {
    return cluster.empty() ? 0 : cluster_dist(inst->get_location(), cluster);
  });

  // sort clusters by distance
  std::vector<size_t> cluster_id_list(clusters.size());
  std::iota(cluster_id_list.begin(), cluster_id_list.end(), 0);
  std::ranges::sort(cluster_id_list, [&](const int& a, const int& b) { return dist[a] < dist[b]; });

  // new_cluster_id is random choose from the [empty_cluster (Up to 1), contain_cluster (All), closest_cluster (Up to 4)]
  auto empty_cluster_count = 0;
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) { empty_cluster_count += cluster.empty() ? 1 : 0; });
  auto contain_cluster_count = 0;
  std::ranges::for_each(dist, [&](const double& d) { contain_cluster_count += d == 0 ? 1 : 0; });

  size_t start = empty_cluster_count > 0 ? empty_cluster_count - 1 : 0;
  // size_t end = start + contain_cluster_count + static_cast<size_t>(empty_cluster_count > 0);
  size_t end = start + 1;
  size_t new_cluster_id = 0;
  if (start < end) {
    std::uniform_int_distribution<size_t> new_cluster_dist(start, end);
    new_cluster_id = cluster_id_list[new_cluster_dist(_gen)];
    while (new_cluster_id == cluster_id) {
      new_cluster_id = cluster_id_list[new_cluster_dist(_gen)];
    }
  }
  return new_cluster_id;
}
/**
 * @brief find instance's id which is close to the boundary
 *
 * @param cluster
 * @return std::vector<size_t>
 */
std::vector<size_t> AnnealOptInterface::findBoundId(const std::vector<Pin*>& cluster)
{
  std::vector<Point> points;
  std::ranges::transform(cluster, std::back_inserter(points), [](const Pin* inst) { return inst->get_location(); });

  auto bound = points;
  BalanceClustering::convexHull(bound);

  auto is_on_bound = [&](const Pin* inst, std::vector<Point>& boundary) {
    auto loc = inst->get_location();
    auto x = loc.x();
    auto y = loc.y();
    for (size_t i = 0; i < boundary.size(); ++i) {
      auto start = boundary[i];
      auto end = boundary[(i + 1) % boundary.size()];
      auto cross = (end.x() - start.x()) * (y - start.y()) - (end.y() - start.y()) * (x - start.x());
      if (cross == 0) {
        return true;
      }
    }
    return false;
  };

  points.clear();
  std::unordered_set<size_t> bound_id;
  for (size_t i = 0; i < cluster.size(); ++i) {
    auto* inst = cluster[i];
    if (is_on_bound(inst, bound)) {
      bound_id.insert(i);
    } else {
      points.push_back(inst->get_location());
    }
  }
  // // second convex hull opt
  // if (_no_change > 15 && bound_id.size() < cluster.size()) {
  //   // remove bound points in "points"
  //   auto bound = points;
  //   BalanceClustering::convexHull(bound);
  //   for (size_t i = 0; i < cluster.size(); ++i) {
  //     auto* inst = cluster[i];
  //     if (is_on_bound(inst, bound)) {
  //       bound_id.insert(i);
  //     }
  //   }
  // }

  return std::vector<size_t>(bound_id.begin(), bound_id.end());
}
/**
 * @brief init the cost map
 *
 */
void AnnealOptInterface::initCostMap()
{
  _cost_map.resize(_cur_solution.size());
  for (size_t i = 0; i < _cur_solution.size(); ++i) {
    auto cluster = _cur_solution[i];
    auto* net = buildNet(cluster);
    auto net_cost = cost(net);
    TimingPropagator::resetNet(net);
    _cost_map[i] = net_cost;
    _sorted_cluster_id_map.insert({ClusterState{cluster.size(), net_cost}, i});
  }
  _cur_cost = std::accumulate(_cost_map.begin(), _cost_map.end(), 0.0);
}
/**
 * @brief update the cost map
 *
 * @param op
 */
void AnnealOptInterface::updateCostMap(const Operation& op)
{
  auto cluster_id = op.cluster_id;
  auto neighbor_id = op.neighbor_id;
  _cost_map[cluster_id] = op.from_cost;
  _cost_map[neighbor_id] = op.to_cost;
  // remove multimap which value is cluster_id or neighbor_id
  for (auto it = _sorted_cluster_id_map.begin(); it != _sorted_cluster_id_map.end();) {
    if (it->second == cluster_id || it->second == neighbor_id) {
      it = _sorted_cluster_id_map.erase(it);
    } else {
      ++it;
    }
  }
  _sorted_cluster_id_map.insert({ClusterState{_cur_solution[cluster_id].size(), op.from_cost}, cluster_id});
  _sorted_cluster_id_map.insert({ClusterState{_cur_solution[neighbor_id].size(), op.to_cost}, neighbor_id});
}
/**
 * @brief center of the cluster
 *
 * @param cluster
 * @return Point
 */
Point AnnealOptInterface::center(const std::vector<Pin*>& cluster)
{
  int64_t x = 0;
  int64_t y = 0;
  x = std::accumulate(cluster.begin(), cluster.end(), x, [](int64_t total, const Pin* inst) { return total + inst->get_location().x(); });
  y = std::accumulate(cluster.begin(), cluster.end(), y, [](int64_t total, const Pin* inst) { return total + inst->get_location().y(); });

  return Point(x / cluster.size(), y / cluster.size());
}
/**
 * @brief build net for the cluster
 *
 * @param cluster
 * @return Net*
 */
Net* AnnealOptInterface::buildNet(const std::vector<Pin*>& cluster)
{
  if (cluster.empty()) {
    return nullptr;
  }
  std::vector<Pin*> load_pins = cluster;
  TreeBuilder::localPlace(load_pins);
  auto* temp_buf = TreeBuilder::defaultTree("temp", load_pins, TimingPropagator::getSkewBound(), std::nullopt, TopoType::kBiPartition);
  temp_buf->set_cell_master(TimingPropagator::getMinSizeCell());

  auto* temp_net = TimingPropagator::genNet("temp", temp_buf->get_driver_pin(), load_pins);
  TimingPropagator::update(temp_net);
  return temp_net;
}
/**
 * @brief build nets for the clusters
 *
 * @param clusters
 * @return std::vector<Net*>
 */
std::vector<Net*> AnnealOptInterface::buildNets(const std::vector<std::vector<Pin*>>& clusters)
{
  std::vector<Net*> temp_nets;
  // for all not empty clusters, build salt net
  std::ranges::for_each(clusters, [&](const std::vector<Pin*>& cluster) {
    if (cluster.empty()) {
      return;
    }
    auto* temp_net = buildNet(cluster);
    temp_nets.push_back(temp_net);
  });
  return temp_nets;
}
/**
 * @brief latency cost of the clusters
 *
 * @param net
 * @return double
 */
double LatAnnealOpt::cost(Net* net)
{
  if (net == nullptr) {
    return 0;
  }
  auto* driver_pin = net->get_driver_pin();
  return driver_pin->get_max_delay() * _correct_coef;
}
/**
 * @brief init the Violation solver parameters
 *
 * @param max_iter
 * @param cooling_rate
 * @param temperature
 */
void VioAnnealOpt::initParameter(const size_t& max_iter, const double& cooling_rate, const double& temperature)
{
  AnnealOptInterface::initParameter(max_iter, cooling_rate, temperature);
  _max_fanout = TimingPropagator::getMaxFanout();
  _max_cap = TimingPropagator::getMaxCap();
  _max_net_len = TimingPropagator::getMaxLength();
  _skew_bound = TimingPropagator::getSkewBound();
}
/**
 * @brief init the VioAnnealOpt solver parameters
 *
 * @param max_iter
 * @param cooling_rate
 * @param temperature
 * @param max_fanout
 * @param max_cap
 * @param max_net_len
 * @param skew_bound
 */
void VioAnnealOpt::initParameter(const size_t& max_iter, const double& cooling_rate, const double& temperature, const int& max_fanout,
                                 const double& max_cap, const double& max_net_len, const double& skew_bound)
{
  initParameter(max_iter, cooling_rate, temperature);
  _max_fanout = max_fanout;
  _max_cap = max_cap;
  _max_net_len = max_net_len;
  _skew_bound = skew_bound;
}
/**
 * @brief violation cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::cost(Net* net)
{
  if (net == nullptr) {
    return 0;
  }
  return _correct_coef
         * (_cap_coef * capVioCost(net) + _wirelength_coef * wireLengthVioCost(net) + _skew_coef * skewVioCost(net)
            + _skew_coef * slewCost(net) + _fanout_coef * fanoutVioCost(net));
}
/**
 * @brief design cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::designCost(const Net* net)
{
  return capCost(net) + wireLengthCost(net) + skewCost(net) + levelCapLoadCost(net);
}
/**
 * @brief wire length cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::wireLengthCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto wire_length = driver_pin->get_sub_len();
  return wire_length * TimingPropagator::getUnitCap();
}
/**
 * @brief wire length violation cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::wireLengthVioCost(const Net* net)
{
  double wire_cost = 0.0;
  auto* driver_pin = net->get_driver_pin();
  auto wire_length = driver_pin->get_sub_len();
  if (wire_length > _max_net_len) {
    wire_cost += (wire_length - _max_net_len) * TimingPropagator::getUnitCap();
  }
  return wire_cost;
}
/**
 * @brief cap load of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::capCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  return driver_pin->get_cap_load();
}
/**
 * @brief cap load violation of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::capVioCost(const Net* net)
{
  double cap_cost = 0.0;
  auto* driver_pin = net->get_driver_pin();
  auto cap_load = driver_pin->get_cap_load();
  if (cap_load > _max_cap) {
    cap_cost += cap_load - _max_cap;
  }
  return cap_cost;
}
/**
 * @brief fanout violation cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::fanoutVioCost(const Net* net)
{
  double fanout_cost = 0.0;
  auto fanout = net->getFanout();
  if (fanout > _max_fanout) {
    fanout_cost += (fanout - _max_fanout) * _max_cap;
  }
  if (fanout == 1) {
    fanout_cost += _max_cap;
  }
  return fanout_cost;
}
/**
 * @brief latency cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::latencyCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto latency = driver_pin->get_max_delay();
  auto cost = latency / TimingPropagator::getUnitRes();
  return cost;
}
/**
 * @brief skew cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::skewCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto skew = driver_pin->get_max_delay() - driver_pin->get_min_delay();
  auto cost = skew / TimingPropagator::getUnitRes();
  return cost;
}
/**
 * @brief skew violation cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::skewVioCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto skew = driver_pin->get_max_delay() - driver_pin->get_min_delay();
  double cost = std::max(0.0, skew - _skew_bound) / TimingPropagator::getUnitRes();
  return cost;
}
/**
 * @brief slew cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::slewCost(const Net* net)
{
  double cost = 0;
  auto* driver_pin = net->get_driver_pin();
  driver_pin->preOrder([&](Node* node) {
    auto slew = node->get_slew_in();
    cost += node->isSinkPin() ? std::max(0.0, slew - TimingPropagator::getMaxSinkTran())
                              : std::max(0.0, slew - TimingPropagator::getMaxBufTran());
  });
  cost /= TimingPropagator::getUnitRes() * TimingPropagator::getUnitCap();
  cost = std::sqrt(cost);
  cost *= TimingPropagator::getUnitCap();
  return cost;
}
/**
 * @brief level cap load cost of the net
 *
 * @param net
 * @return double
 */
double VioAnnealOpt::levelCapLoadCost(const Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto* inst = driver_pin->get_inst();
  return inst->get_load_pin()->get_cap_load();
}

}  // namespace icts