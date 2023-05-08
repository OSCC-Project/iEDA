#pragma once

#include "RTU.hpp"

namespace irt {

class RAConfig
{
 public:
  RAConfig() = default;
  ~RAConfig() = default;

  std::string temp_directory_path;
  irt_int bottom_routing_layer_idx;
  irt_int top_routing_layer_idx;
  std::map<irt_int, double> layer_idx_utilization_ratio;
  double initial_penalty;
  double penalty_drop_rate;
  irt_int outer_iter_num;
  irt_int inner_iter_num;
};

}  // namespace irt
