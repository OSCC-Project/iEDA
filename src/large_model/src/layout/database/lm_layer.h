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
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <unordered_map>
#include <vector>

#include "lm_layer_grid.h"
#include "lm_net.h"

namespace ilm {

class LmLayoutLayer
{
 public:
  LmLayoutLayer() {}
  ~LmLayoutLayer() {}
    // 添加移动构造函数
  LmLayoutLayer(LmLayoutLayer&& other) noexcept
      : _layer_name(std::move(other._layer_name)),
        _wire_width(other._wire_width),
        _b_routing(other._b_routing),
        _b_horizontal(other._b_horizontal),
        _layer_order(other._layer_order),
        _llx(other._llx),
        _lly(other._lly),
        _urx(other._urx),
        _ury(other._ury),
        _row_num(other._row_num),
        _row_space(other._row_space),
        _col_num(other._col_num),
        _col_space(other._col_space),
        _grid(std::move(other._grid)),
        _net_map(std::move(other._net_map))
  {
  }

  // 添加移动赋值运算符
  LmLayoutLayer& operator=(LmLayoutLayer&& other) noexcept {
      if (this != &other) {
          _layer_name = std::move(other._layer_name);
          _wire_width = other._wire_width;
          _b_routing = other._b_routing;
          _b_horizontal = other._b_horizontal;
          _layer_order = other._layer_order;
          _llx = other._llx;
          _lly = other._lly;
          _urx = other._urx;
          _ury = other._ury;
          _row_num = other._row_num;
          _row_space = other._row_space;
          _col_num = other._col_num;
          _col_space = other._col_space;
          _grid = std::move(other._grid);
          _net_map = std::move(other._net_map);
      }
      return *this;
  }

  // getter
  std::string& get_layer_name() { return _layer_name; }
  int get_wire_width() { return _wire_width; }
  bool is_routing() { return _b_routing; }
  LmLayerGrid& get_grid() { return _grid; }

  std::map<int, LmNet>& get_net_map() { return _net_map; }
  LmNet* get_net(int net_id);
  int get_layer_order() { return _layer_order; }
  bool is_horizontal() { return _b_horizontal; }

  // setter
  void set_layer_name(std::string name) { _layer_name = name; }
  void set_wire_width(int wire_width) { _wire_width = wire_width; }
  void set_as_routing(bool b_routing) { _b_routing = b_routing; }
  void set_layer_order(int order) { _layer_order = order; }
  void set_llx(int value) { _llx = value; }
  void set_lly(int value) { _lly = value; }
  void set_urx(int value) { _urx = value; }
  void set_ury(int value) { _ury = value; }
  void set_horizontal(bool b_horizontal) { _b_horizontal = b_horizontal; }

  // operator
  LmNet* getOrCreateNet(int net_id);

 private:
  std::string _layer_name;
  int _wire_width = 0;
  bool _b_routing;
  bool _b_horizontal;
  int _layer_order;
  int _llx;
  int _lly;
  int _urx;
  int _ury;
  int _row_num;  /// row number
  int _row_space;
  int _col_num;  /// col number
  int _col_space;
  LmLayerGrid _grid;
  std::map<int, LmNet> _net_map;
};

class LmLayoutLayers
{
 public:
  LmLayoutLayers() {};
  ~LmLayoutLayers() {}

  // getter
  int get_layer_order_top() { return _layer_order_top; }
  int get_layer_order_bottom() { return _layer_order_bottom; }
  std::unordered_map<int, LmLayoutLayer>& get_layout_layer_map() { return _layout_layers; }
  LmLayoutLayer* findLayoutLayer(int order);

  // setter
  void set_layer_order_top(int order) { _layer_order_top = order; }
  void set_layer_order_bottom(int order) { _layer_order_bottom = order; }

  // operator

 private:
  int _layer_order_top = -1;
  int _layer_order_bottom = -1;
  std::unordered_map<int, LmLayoutLayer> _layout_layers;  /// int : layer order
};

}  // namespace ilm
