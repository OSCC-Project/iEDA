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
 * @author Jianrong Su
 * @brief Grid manager for PDN
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "SingleTemplate.hh"
#include "TemplateLib.hh"

namespace ipnp {

class PNPConfig;  // Forward declaration

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

  // Getters
  double get_height() { return _y_right_top - _y_left_bottom; }
  double get_width() { return _x_right_top - _x_left_bottom; }
  std::pair<double, double> get_left_bottom_coordinate() const
  {
    std::pair<double, double> left_bottom_coordinate(_x_left_bottom, _y_left_bottom);
    return left_bottom_coordinate;
  }
  std::pair<double, double> get_right_top_coordinate() const
  {
    std::pair<double, double> right_top_coordinate(_x_right_top, _y_right_top);
    return right_top_coordinate;
  }

  // Setters
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
   GridManager() = default;
  ~GridManager() = default;

  // Getters
  const std::vector<int>& get_power_layers() const { return _power_layers; }
  int get_layer_count() const { return _layer_count; }
  int get_ho_region_num() const { return _ho_region_num; }
  int get_ver_region_num() const { return _ver_region_num; }
  int32_t get_core_llx() const { return _core_llx; }
  int32_t get_core_lly() const { return _core_lly; }
  int32_t get_core_urx() const { return _core_urx; }
  int32_t get_core_ury() const { return _core_ury; }
  double get_core_width() const { return _core_width; }
  double get_core_height() const { return _core_height; }
  const auto& get_grid_data() const { return _grid_data; }
  const auto& get_template_data() const { return _template_data; }
  const std::vector<SingleTemplate>& get_horizontal_templates() const { return _template_libs.get_horizontal_templates(); }
  const std::vector<SingleTemplate>& get_vertical_templates() const { return _template_libs.get_vertical_templates(); }
  const TemplateLib& get_template_libs() const { return _template_libs; }

  // Setters
  void set_power_layers(std::vector<int> power_layers){
    _power_layers = power_layers;
    _layer_count = power_layers.size();
  }
  void set_layer_count(int layer_count) { _layer_count = layer_count; }
  void set_ho_region_num(int ho_region_num) { _ho_region_num = ho_region_num; }
  void set_ver_region_num(int ver_region_num) { _ver_region_num = ver_region_num; }
  void set_core_llx(int32_t core_llx) { _core_llx = core_llx; }
  void set_core_lly(int32_t core_lly) { _core_lly = core_lly; }
  void set_core_urx(int32_t core_urx) { _core_urx = core_urx; }
  void set_core_ury(int32_t core_ury) { _core_ury = core_ury; }
  void set_core_width(double chip_width) { _core_width = chip_width; }
  void set_core_height(double chip_height) { _core_height = chip_height; }
  void set_die_width(double die_width) { _die_width = die_width; }
  void set_die_height(double die_height) { _die_height = die_height; }
  void set_grid_data(std::vector<std::vector<std::vector<PDNRectanGridRegion>>> grid_data) { _grid_data = grid_data; }
  void set_single_template(int layer_idx, int row, int col, const SingleTemplate& single_template) { _template_data[layer_idx][row][col] = single_template; }

  // Initialize with default templates
  void init_GridManager_data();
  
  // Initialize with templates from configuration
  void init_GridManager_data(const PNPConfig* config);
  
private:
  std::vector<int> _power_layers;   // layers that have power nets
  int _layer_count;     // total number of layers
  int _ho_region_num;   // number of horizontal regions
  int _ver_region_num;  // number of vertical regions

  int32_t _core_width;   // width of the core
  int32_t _core_height;  // height of the core
  int32_t _die_width;   // width of the core
  int32_t _die_height;  // height of the core
  int32_t _core_llx;    // left bottom x coordinate of the core 
  int32_t _core_lly;    // left bottom y coordinate of the core
  int32_t _core_urx;    // right top x coordinate of the core
  int32_t _core_ury;    // right top y coordinate of the core

  std::vector<std::pair<std::string, std::string>> _power_nets;  // only VDD VSS / VDD GND
  std::vector<std::vector<std::vector<PDNRectanGridRegion>>> _grid_data;      // [layer][row][col]
  std::vector<std::vector<std::vector<SingleTemplate>>> _template_data;       // [layer][row][col]
  TemplateLib _template_libs;   // Template library manager

  void initialize_grid_data(int32_t width, int32_t height);
};

}  // namespace ipnp
