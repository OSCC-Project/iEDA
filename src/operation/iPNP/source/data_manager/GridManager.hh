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
  PDNGridTemplate();
  ~PDNGridTemplate() = default;

  struct GridPerLayer
  {
    std::string direction = "horizontal";  // horizontal, vertical
    double offset = 1.0;
    double width = 5.0;
    double space = 5.0;
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
  PDNGridRegion();
  ~PDNGridRegion() = default;

 private:
  std::string _type;  // e.g. rectangle, irregular(coverd by Macro)
  // std::vector<int> _region_position;  // e.g. {0,0}, {0,1}...{1,0}, {1,1}
  //                                     // If GridManager contains GridRegion position information, this member variable can be omitted.
};

class PDNRectanGridRegion : public PDNGridRegion
{
 public:
  PDNRectanGridRegion();
  ~PDNRectanGridRegion() = default;

  double get_height() { return _height; }
  double get_width() { return _width; }

  void set_height(double height) { _height = height; }
  void set_width(double width) { _width = width; }

 private:
  /**
   *don't need position information, which is included in GridManager.
   */
  // double x_left_bottom;
  // double y_left_bottom;
  // double x_right_top;
  // double y_right_top;
  double _height;
  double _width;
};

// class GridNetwork{
//  private:
//   int ho_region_num, ver_region_num;
//   std::vector<std::vector<PDNGridRegion>> _grid_data; //which GridRegion is on position[][].
//   std::vector<std::vector<int>> _template_data; //which GridTemplate is on position[][]. Just need the template number.
// };

class GridManager
{
 public:
  GridManager();
  ~GridManager() = default;

  int get_ho_region_num() { return _ho_region_num; }
  int get_ver_region_num() { return _ver_region_num; }
  double get_chip_width() { return _chip_width; }
  double get_chip_height() { return _chip_height; }
  // std::vector<std::vector<PDNGridRegion>> get_grid_data() { return _grid_data; }
  auto& get_grid_data() { return _grid_data; }
  std::vector<std::vector<int>> get_template_data() { return _template_data; }
  std::vector<PDNGridTemplate> get_template_libs() { return _template_libs; }

  void set_ho_region_num(int ho_region_num) { _ho_region_num = ho_region_num; }
  void set_ver_region_num(int ver_region_num) { _ver_region_num = ver_region_num; }
  void set_chip_width(double chip_width) { _chip_width = chip_width; }
  void set_chip_height(double chip_height) { _chip_height = chip_height; }
  void set_grid_data(std::vector<std::vector<PDNRectanGridRegion>> grid_data) { _grid_data = grid_data; }
  void set_template_data(std::vector<std::vector<int>> template_data) { _template_data = template_data; }
  void set_template_libs(std::vector<PDNGridTemplate> template_libs) { _template_libs = template_libs; }

 private:
  int _ho_region_num;
  int _ver_region_num;
  double _chip_width;
  double _chip_height;
  // std::vector<std::vector<PDNGridRegion>> _grid_data;  // which GridRegion is on position[][].
  std::vector<std::vector<PDNRectanGridRegion>> _grid_data;  // which GridRegion is on position[][].
  std::vector<std::vector<int>> _template_data;              // which GridTemplate is on position[][].
  std::vector<PDNGridTemplate> _template_libs;               // Starting from 1
};

}  // namespace ipnp
