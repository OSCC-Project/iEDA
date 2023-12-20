/**
 * @file SAEvaluateWl.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-7
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SA_EVALUATE_H
#define IMP_SA_EVALUATE_H

#include <map>
#include <vector>

#include "../../utility/logger/Logger.hpp"
#include "SAPlacement.hh"
namespace imp {

template <typename CoordType, typename RepresentType>
struct SAEvaluate
{
 public:
  SAEvaluate() = default;
  ~SAEvaluate() = default;
  SAEvaluate(const SAEvaluate& other) = default;

  const std::map<std::string, double>& get_cost_avg_dict() const { return _cost_avg_dict; }
  const std::map<std::string, double>& get_cost_delta_dict() const { return _cost_delta_dict; }
  const std::map<std::string, double>& get_cost_std_variance_dict() const { return _cost_std_variance_dict; }
  virtual std::map<std::string, double> calCostDict(SAPlacement<CoordType, RepresentType>& placement) = 0;

  // double getNormedCost(SAPlacement<CoordType, RepresentType>& placement)
  // {
  //   double cost = getCost(placement);
  //   if (_is_normed) {
  //     cost = (cost - _avg_cost) / _std_variance;
  //   }
  //   placement.set_cost(cost);
  //   return cost;
  // }

  void initCostNorm(SAPlacement<CoordType, RepresentType>& placement, size_t perturb_steps = 200)
  {
    auto placement_cp = placement;
    std::map<std::string, std::vector<double>> cost_trajectory;
    // perturb n steps
    std::map<std::string, double> cost_dict;
    for (size_t i = 0; i < perturb_steps; ++i) {
      placement_cp.randomizeRepresentation();
      cost_dict = calCostDict(placement_cp);
      for (auto& [cost_name, cost_value] : cost_dict) {
        cost_trajectory[cost_name].push_back(cost_value);
      }
    }

    std::vector<double> normed_value;
    for (auto& [cost_name, trajectory] : cost_trajectory) {
      normed_value = calNormValue(trajectory);
      _cost_avg_dict[cost_name] = (normed_value[0]);
      _cost_delta_dict[cost_name] = (normed_value[1]);
      _cost_std_variance_dict[cost_name] = (normed_value[2]);
    }
    _is_normed = true;
  }

 protected:
  std::map<std::string, double> normalizeCostDict(const std::map<std::string, double>& cost_dict)
  {
    if (!_is_normed) {
      ERROR("Norm value not initialized!");
    }
    std::map<std::string, double> normed_cost_dict;
    for (const auto& [cost_name, cost_value] : cost_dict) {
      normed_cost_dict[cost_name] = cost_value - _cost_avg_dict[cost_name] / _cost_std_variance_dict[cost_name];
    }
    return normed_cost_dict;
  }

 private:
  std::map<std::string, double> _cost_avg_dict;
  std::map<std::string, double> _cost_delta_dict;
  std::map<std::string, double> _cost_std_variance_dict;
  bool _is_normed = false;

  std::vector<double> calNormValue(const std::vector<double>& cost_list)
  {
    size_t size = cost_list.size();
    double avg_cost = cost_list[0];
    double delta_cost = 0, std_variance = 0;
    for (size_t i = 1; i < size; ++i) {
      avg_cost += cost_list[i];
      delta_cost += std::abs(cost_list[i] - cost_list[i - 1]);
    }
    avg_cost /= size;
    delta_cost /= size - 1;
    for (size_t i = 0; i < size; ++i) {
      std_variance += std::pow(cost_list[i] - avg_cost, 2);
    }
    std_variance = std::sqrt(std_variance / size);
    return {avg_cost, delta_cost, std_variance};
  }
};

}  // namespace imp

#endif