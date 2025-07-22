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
#pragma once

#include "RTHeader.hpp"
#include "SAComParam.hpp"

namespace irt {

class SAModel
{
 public:
  SAModel() = default;
  ~SAModel() = default;
  // getter
  SAComParam& get_sa_com_param() { return _sa_com_param; }
  std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>>& get_grid_pair_list_list() { return _grid_pair_list_list; }
  // setter
  void set_sa_com_param(const SAComParam& sa_com_param) { _sa_com_param = sa_com_param; }
  void set_grid_pair_list_list(const std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>>& grid_pair_list_list)
  {
    _grid_pair_list_list = grid_pair_list_list;
  }

 private:
  SAComParam _sa_com_param;
  std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>> _grid_pair_list_list;
};

}  // namespace irt
