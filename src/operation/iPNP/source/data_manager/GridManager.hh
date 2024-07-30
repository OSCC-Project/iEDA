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
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <map>
#include <string>
#include <vector>

namespace ipnp {

enum class GridRegionShape
{
  rectangle,
  irregular  // caused by macro block
};

enum class PowerType
{
  VDD,
  VSS,
};

enum class StripeDirection
{
  horizontal,
  vertical
};

/**
 * @brief a single layer Template in 3D Template block
 */
class SingleLayerGrid
{
 public:
  SingleLayerGrid(StripeDirection direction = StripeDirection::horizontal, PowerType first_stripe_power_type = PowerType::VDD,
                  double width = 2.0, double pg_offset = 3.0, double space = 10.0, double offset = 1.0);
  ~SingleLayerGrid() = default;

  StripeDirection get_direction() { return _direction; }
  PowerType get_first_stripe_power_type() { return _first_stripe_power_type; }
  double get_width() { return _width; }
  double get_pg_offset() { return _pg_offset; }
  double get_space() { return _space; }
  double get_offset() { return _offset; }

  void set_direction(StripeDirection direction) { _direction = direction; }
  void set_first_stripe_power_type(PowerType first_stripe_power_type) { _first_stripe_power_type = first_stripe_power_type; }
  void set_width(double width) { _width = width; }
  void set_pg_offset(double pg_offset) { _pg_offset = pg_offset; }
  void set_space(double space) { _space = space; }
  void set_offset(double offset) { _offset = offset; }

 private:
  StripeDirection _direction = StripeDirection::horizontal;
  PowerType _first_stripe_power_type = PowerType::VDD;
  /**
   * @attention DRC: width + pg_offset < space
   */
  double _width = 2.0;
  double _pg_offset = 3.0;  // offset between the first power and ground wire
  double _space = 10.0;     // distance between edges of two VDD wire
  double _offset = 1.0;     // if direction is horizontal, offset from bottom; if direction is vertical, offset from left.
};

/**
 * @brief 3D Template block
 */
class PDNGridTemplate
{
 public:
  PDNGridTemplate();
  ~PDNGridTemplate() = default;

  auto get_layers_occupied() { return _layers_occupied; }
  auto get_grid_per_layer() { return _grid_per_layer; }

 private:
  std::vector<int> _layers_occupied;  // e.g. {1,2,6,7,8,9}
  std::map<int, SingleLayerGrid> _grid_per_layer;
};

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

  double set_x_left_bottom(double x) { _x_left_bottom = x; }
  double set_y_left_bottom(double y) { _y_left_bottom = y; }
  double set_x_right_top(double x) { _x_right_top = x; }
  double set_y_right_top(double y) { _y_right_top = y; }

 private:
  double _x_left_bottom;
  double _y_left_bottom;
  double _x_right_top;
  double _y_right_top;
};

class GridManager
{
 public:
  GridManager();
  ~GridManager() = default;

  int get_ho_region_num() { return _ho_region_num; }
  int get_ver_region_num() { return _ver_region_num; }
  double get_chip_width() { return _chip_width; }
  double get_chip_height() { return _chip_height; }
  auto get_grid_data() { return _grid_data; }
  auto get_template_data() { return _template_data; }
  auto get_template_libs() { return _template_libs; }

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

  std::vector<std::pair<std::string, std::string>> _power_nets;  // only VDD VSS / VDD GND
  std::vector<std::vector<PDNRectanGridRegion>> _grid_data;      // which GridRegion is on position[][].
  std::vector<std::vector<int>> _template_data;                  // which GridTemplate is on position[][].
  std::vector<PDNGridTemplate> _template_libs;                   // Starting from 1
};

}  // namespace ipnp
