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
 * @file AnnealOpt.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <limits>
#include <map>
#include <random>
#include <ranges>

#include "Pin.hh"

namespace icts {
enum class AnnealOptType
{
  kLatencyCost,
  kViolationCost,
};
struct Operation
{
  size_t cluster_id;
  size_t neighbor_id;
  size_t pin_id;
  double from_cost;
  double to_cost;
};
struct ClusterState
{
  size_t size;
  double cost;
};
struct ClusterStateCompare
{
  bool operator()(const ClusterState& lhs, const ClusterState& rhs) const
  {
    auto size_rhs = rhs.size;
    if (size_rhs == 0) {
      return true;
    }
    auto size_lhs = lhs.size;
    if (size_lhs == 0) {
      return false;
    }
    return lhs.cost > rhs.cost;
  }
};
/**
 * @brief AnnealOpt for rebalance clustering
 *       input: clusters
 *       constraint: net_num, max_fanout, max_cap, max_net_len,
 *                   p(coefficient of net length), q(coefficient of net skew)
 *       output: new clusters which satisfy the constraints
 *
 */
class AnnealOptInterface
{
 public:
  AnnealOptInterface(const std::vector<std::vector<Pin*>>& clusters) : _cur_solution(clusters){};
  ~AnnealOptInterface() = default;
  void initParameter(const size_t& max_iter, const double& cooling_rate, const double& temperature)
  {
    _max_iter = max_iter;
    _cooling_rate = cooling_rate;
    _temperature = temperature;
  }
  // run
  void automaticTemperature();
  std::vector<std::vector<Pin*>> run(const bool& log = false);

  double get_best_cost() const { return _best_cost; };
  double get_improve() const { return _improve; };

 protected:
  void updateSolution(const std::vector<std::vector<Pin*>>& new_solution, const Operation& op);
  /**
   * @brief random operation
   *
   */

  std::vector<std::vector<Pin*>> commitOperation(Operation& op);

  std::vector<std::vector<Pin*>> randomSwap(const std::vector<std::vector<Pin*>>& clusters);

  Operation randomMove(const std::vector<std::vector<Pin*>>& clusters);

  size_t randomChooseCluster(const std::vector<std::vector<Pin*>>& clusters, const double& ratio);

  size_t randomChoosePin(const std::vector<Pin*>& cluster);

  size_t randomChooseNeighbor(const std::vector<std::vector<Pin*>>& clusters, const size_t& cluster_id, const size_t& pin_id);

  std::vector<size_t> findBoundId(const std::vector<Pin*>& clusters);
  /**
   * @brief cost function
   *
   */
  void initCostMap();
  void updateCostMap(const Operation& op);
  Point center(const std::vector<Pin*>& cluster);

  /**
   * @brief Net builder
   *
   */
  Net* buildNet(const std::vector<Pin*>& cluster);
  std::vector<Net*> buildNets(const std::vector<std::vector<Pin*>>& clusters);

  virtual double cost(Net* net) = 0;
  /**
   * @brief Database and parameters
   *
   */
  std::vector<std::vector<Pin*>> _cur_solution;

  std::multimap<ClusterState, size_t, ClusterStateCompare> _sorted_cluster_id_map;

  const std::mt19937::result_type _seed = 0;
  std::mt19937 _gen = std::mt19937(_seed);

  bool _auto_temp = false;

  const double _correct_coef = 1e2;
  size_t _max_iter = 0;
  double _cooling_rate = 0;
  double _temperature = 0;

  std::vector<double> _cost_map;
  double _new_cost = std::numeric_limits<double>::max();
  double _cur_cost = std::numeric_limits<double>::max();
  double _best_cost = std::numeric_limits<double>::max();
  double _improve = 0;
  int _no_change = 0;
};
class LatAnnealOpt : public AnnealOptInterface
{
 public:
  LatAnnealOpt(const std::vector<std::vector<Pin*>>& clusters) : AnnealOptInterface(clusters) { initCostMap(); };
  ~LatAnnealOpt() = default;
  std::vector<std::vector<Pin*>> run(const bool& log = false)
  {
    LOG_INFO_IF(log) << "Begin Anneal Optimization By [Latency Cost]";
    return AnnealOptInterface::run(log);
  }

 private:
  // latency, empty buffering will lead more latency
  double cost(Net* net) override;
};
class VioAnnealOpt : public AnnealOptInterface
{
 public:
  VioAnnealOpt(const std::vector<std::vector<Pin*>>& clusters) : AnnealOptInterface(clusters){};
  ~VioAnnealOpt() = default;
  // init parameters
  void initParameter(const size_t& max_iter, const double& cooling_rate, const double& temperature);
  void initParameter(const size_t& max_iter, const double& cooling_rate, const double& temperature, const int& max_fanout,
                     const double& max_cap, const double& max_net_len, const double& skew_bound);

  std::vector<std::vector<Pin*>> run(const bool& log = false)
  {
    LOG_INFO_IF(log) << "Begin Anneal Optimization By [Violation Cost]";
    return AnnealOptInterface::run(log);
  }

 private:
  // violation, convert all violation to wirelength
  double cost(Net* net) override;
  double designCost(const Net* net);
  double wireLengthCost(const Net* net);
  double wireLengthVioCost(const Net* net);
  double capCost(const Net* net);
  double capVioCost(const Net* net);
  double fanoutVioCost(const Net* net);
  double latencyCost(const Net* net);
  double skewCost(const Net* net);
  double skewVioCost(const Net* net);
  double slewCost(const Net* net);
  double levelCapLoadCost(const Net* net);

  int _max_fanout = 0;
  double _max_cap = 0;
  double _max_net_len = 0;
  double _skew_bound = 0;

  const double _cap_coef = 1;
  const double _fanout_coef = 10;
  const double _skew_coef = 1;
  const double _wirelength_coef = 1;
};
}  // namespace icts