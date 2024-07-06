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

#include <map>
#include <string>
#include <vector>

namespace ipnp {

// 3D Template block
class PDNGridTemplate
{
 public:
  struct GridPerLayer
  {
    std::string direction;
    double offset, width, space;
  };

 private:
  std::vector<int> _layers_occupied;           // e.g. {1,2,6,7,8,9}
  std::map<int, GridPerLayer> _layer_to_grid;  // 2D network per layer
};

// Base class
class PDNGridRegion
{
 public:
  // TODO: Constructor function; get function...
 private:
  std::string _type;  // e.g. rectangle, irregular(coverd by Macro)
  std::vector<int> _region_position;  // e.g. {0,0}, {0,1}...{1,0}, {1,1}
                                      // If GridManager contains GridRegion position information, this member variable can be omitted.
};

class PDNRectanGridRegion : public PDNGridRegion
{
 private:
  double x_left_bottom;
  double y_left_bottom;
  double x_right_top;
  double y_right_top;
};

// class GridNetwork{
//  private://public? 方便GridManager调用?
//   int ho_region_num, ver_region_num;
//   std::vector<std::vector<PDNGridRegion>> _grid_data; //which GridRegion is on position[][].
//   std::vector<std::vector<int>> _template_data; //which GridTemplate is on position[][]. Just need the template number.
// };

class GridManager
{
 public:
  int get_ho_region_num() { return _ho_region_num; }
  int get_ver_region_num() { return _ver_region_num; }
  //std::vector<std::vector<PDNGridRegion>> get_grid_data() { return _grid_data; }
  auto &get_grid_data() { return _grid_data; }
  std::vector<std::vector<int>> get_template_data() { return _template_data; }
  std::vector<PDNGridTemplate> get_template_libs() { return _template_libs; }

  void set_ho_region_num(int ho_region_num) { _ho_region_num = ho_region_num; }
  void set_ver_region_num(int ver_region_num) { _ver_region_num = ver_region_num; }
  void set_grid_data(std::vector<std::vector<PDNGridRegion>> grid_data) { _grid_data = grid_data; }
  void set_template_data(std::vector<std::vector<int>> template_data) { _template_data = template_data; }
  void set_template_libs(std::vector<PDNGridTemplate> template_libs) { _template_libs = template_libs; }

 private:
  int _ho_region_num;
  int _ver_region_num;
  std::vector<std::vector<PDNGridRegion>> _grid_data;  // which GridRegion is on position[][].
  std::vector<std::vector<int>> _template_data;        // which GridTemplate is on position[][]. Just need the template number.
  std::vector<PDNGridTemplate> _template_libs;
};

}  // namespace ipnp
