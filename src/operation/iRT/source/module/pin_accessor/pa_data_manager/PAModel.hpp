#pragma once

#include <vector>

#include "PAGCell.hpp"
#include "PAModelStat.hpp"

namespace irt {

class PAModel
{
 public:
  PAModel() = default;
  ~PAModel() = default;
  // getter
  std::vector<GridMap<PAGCell>>& get_layer_gcell_map() { return _layer_gcell_map; }
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  PAModelStat& get_pa_mode_stat() { return _pa_mode_stat; }
  // setter
  void set_layer_gcell_map(const std::vector<GridMap<PAGCell>>& layer_gcell_map) { _layer_gcell_map = layer_gcell_map; }
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_pa_mode_stat(const PAModelStat& pa_mode_stat) { _pa_mode_stat = pa_mode_stat; }

 private:
  std::vector<GridMap<PAGCell>> _layer_gcell_map;
  std::vector<PANet> _pa_net_list;
  PAModelStat _pa_mode_stat;
};

}  // namespace irt
