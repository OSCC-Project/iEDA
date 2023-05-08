#pragma once

#include "DRModelStat.hpp"
#include "GridMap.hpp"

namespace irt {

class DRModel
{
 public:
  DRModel() = default;
  ~DRModel() = default;
  // getter
  std::vector<DRNet>& get_dr_net_list() { return _dr_net_list; }
  GridMap<DRBox>& get_dr_box_map() { return _dr_box_map; }
  DRModelStat& get_dr_model_stat() { return _dr_model_stat; }
  // setter
  void set_dr_net_list(const std::vector<DRNet>& dr_net_list) { _dr_net_list = dr_net_list; }

 private:
  std::vector<DRNet> _dr_net_list;
  GridMap<DRBox> _dr_box_map;
  DRModelStat _dr_model_stat;
};

}  // namespace irt
