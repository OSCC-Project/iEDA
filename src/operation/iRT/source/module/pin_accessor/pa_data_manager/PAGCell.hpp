#pragma once

#include "LayerCoord.hpp"

namespace irt {

class PAGCell : public LayerCoord
{
 public:
  PAGCell() = default;
  ~PAGCell() = default;

  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_blockage_map() { return _net_blockage_map; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<PlanarRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  // function

 private:
  PlanarRect _real_rect;
  std::map<irt_int, std::vector<PlanarRect>> _net_blockage_map;
};

}  // namespace irt
