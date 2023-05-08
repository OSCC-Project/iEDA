#pragma once

#include <string>

#include "RTU.hpp"

namespace irt {

class RAModelStat
{
 public:
  RAModelStat() = default;
  ~RAModelStat() = default;
  // getter
  double get_max_avg_cost() { return _max_avg_cost; }
  std::vector<double>& get_avg_cost_list() { return _avg_cost_list; }
  // setter
  void set_max_avg_cost(const double max_avg_cost) { _max_avg_cost = max_avg_cost; }
  // function

 private:
  double _max_avg_cost;
  std::vector<double> _avg_cost_list;
};

}  // namespace irt
