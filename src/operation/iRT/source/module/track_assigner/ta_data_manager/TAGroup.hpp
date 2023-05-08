#pragma once

#include "LayerCoord.hpp"

namespace irt {

class TAGroup
{
 public:
  TAGroup() = default;
  ~TAGroup() = default;
  // getter
  std::vector<LayerCoord>& get_coord_list() { return _coord_list; }
  // setter
  void set_coord_list(const std::vector<LayerCoord>& coord_list) { _coord_list = coord_list; }
  // function

 private:
  std::vector<LayerCoord> _coord_list;
};

}  // namespace irt
