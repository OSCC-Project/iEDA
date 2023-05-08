#pragma once

#include "RTU.hpp"

namespace irt {

class GRConfig
{
 public:
  GRConfig() = default;
  ~GRConfig() = default;

  std::string temp_directory_path;
  irt_int bottom_routing_layer_idx = -1;
  irt_int top_routing_layer_idx = -1;
  std::map<irt_int, double> layer_idx_utilization_ratio;
};

}  // namespace irt
