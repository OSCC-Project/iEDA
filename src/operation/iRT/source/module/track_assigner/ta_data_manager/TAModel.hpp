#pragma once

#include <vector>

#include "TAModelStat.hpp"
#include "TANet.hpp"
#include "TAPanel.hpp"

namespace irt {

class TAModel
{
 public:
  TAModel() = default;
  ~TAModel() = default;
  // getter
  std::vector<TANet>& get_ta_net_list() { return _ta_net_list; }
  std::vector<std::vector<TAPanel>>& get_layer_panel_list() { return _layer_panel_list; }
  TAModelStat& get_ta_model_stat() { return _ta_model_stat; }
  // setter
  void set_ta_net_list(std::vector<TANet>& ta_net_list) { _ta_net_list = ta_net_list; }

 private:
  std::vector<TANet> _ta_net_list;
  std::vector<std::vector<TAPanel>> _layer_panel_list;
  TAModelStat _ta_model_stat;
};

}  // namespace irt
