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
/**
 * @file GridManager.hh
 * @author Xinhao li
 * @brief Grid manager for PDN
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "SingleTemplate.hh"
#include "TemplateLib.hh"

namespace ipnp {

enum class GridRegionShape
{
  kRectangle,
  kIrregular  // caused by macro block
};

// enum class PowerType
// {
//   kVDD,
//   kVSS,
// };

// enum class StripeDirection
// {
//   kHorizontal,
//   kVertical
// };

/**
 * @brief Bass class. Have derived class representing regions of different shapes.
 */
class PDNGridRegion
{
 public:
  PDNGridRegion();
  ~PDNGridRegion() = default;

  GridRegionShape get_shape() { return _shape; }

 private:
  GridRegionShape _shape;
};

class PDNRectanGridRegion : public PDNGridRegion
{
 public:
  PDNRectanGridRegion();
  ~PDNRectanGridRegion() = default;

  double get_height() { return _y_right_top - _y_left_bottom; }
  double get_width() { return _x_right_top - _x_left_bottom; }
  std::pair<double, double> get_left_bottom_coordinate()
  {
    std::pair<double, double> left_bottom_coordinate(_x_left_bottom, _y_left_bottom);
    return left_bottom_coordinate;
  }
  std::pair<double, double> get_right_top_coordinate()
  {
    std::pair<double, double> right_top_coordinate(_x_right_top, _y_right_top);
    return right_top_coordinate;
  }

  void set_x_left_bottom(double x) { _x_left_bottom = x; }
  void set_y_left_bottom(double y) { _y_left_bottom = y; }
  void set_x_right_top(double x) { _x_right_top = x; }
  void set_y_right_top(double y) { _y_right_top = y; }

 private:
  double _x_left_bottom;
  double _y_left_bottom;
  double _x_right_top;
  double _y_right_top;
};

/**
 * @brief Manager for PDN grid
 */
class GridManager
{
 public:
  GridManager();
  ~GridManager() = default;

  // Getters
  const std::vector<int>& get_power_layers() const { return _power_layers; }
  int get_layer_count() const { return _layer_count; }
  int get_ho_region_num() const { return _ho_region_num; }
  int get_ver_region_num() const { return _ver_region_num; }
  double get_chip_width() const { return _core_width; }
  double get_chip_height() const { return _core_height; }
  const auto& get_grid_data() const { return _grid_data; }
  const auto& get_template_data() const { return _template_data; }
  const std::vector<SingleTemplate>& get_horizontal_templates() const { return _template_libs.get_horizontal_templates(); }
  const std::vector<SingleTemplate>& get_vertical_templates() const { return _template_libs.get_vertical_templates(); }
  const TemplateLib& get_template_libs() const { return _template_libs; }

  // Setters
  void set_power_layers(std::vector<int> power_layers){
    _power_layers = power_layers;
    _layer_count = power_layers.size();
    initialize_grid_data();
  }
  void set_layer_count(int layer_count) { _layer_count = layer_count; }
  void set_ho_region_num(int ho_region_num) { _ho_region_num = ho_region_num; }
  void set_ver_region_num(int ver_region_num) { _ver_region_num = ver_region_num; }
  void set_core_width(double chip_width) { _core_width = chip_width; }
  void set_core_height(double chip_height) { _core_height = chip_height; }
  void set_grid_data(std::vector<std::vector<std::vector<PDNRectanGridRegion>>> grid_data) { _grid_data = grid_data; } 
  void set_single_template(int layer_idx, int row, int col, const SingleTemplate& single_template) { _template_data[layer_idx][row][col] = single_template; }

private:
  std::vector<int> _power_layers;   // layers that have power nets
  int _layer_count;     // total number of layers
  int _ho_region_num;   // number of horizontal regions
  int _ver_region_num;  // number of vertical regions
  double _core_width;   // width of the core
  double _core_height;  // height of the core

  std::vector<std::pair<std::string, std::string>> _power_nets;  // only VDD VSS / VDD GND
  std::vector<std::vector<std::vector<PDNRectanGridRegion>>> _grid_data;      // [layer][row][col]
  std::vector<std::vector<std::vector<SingleTemplate>>> _template_data;       // [layer][row][col]
  TemplateLib _template_libs;   // Template library manager

  void initialize_grid_data();
};

}  // namespace ipnp
