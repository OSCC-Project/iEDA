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
#include <map>
#include <string>
#include <vector>

#include "lm_layer_grid.h"
#include "lm_net.h"
#include "lm_patch.h"

namespace ilm {

class LmPatchLayer
{
 public:
  LmPatchLayer() = default;
  ~LmPatchLayer() = default;

  // getter
  bool is_routing() { return _b_routing; }
  LmLayerGrid& get_grid() { return _grid; }
  std::vector<std::vector<LmPatch>>& get_patch_matrix() { return _patch_matrix; }
  LmPatch& get_patch(int row_id, int col_id);
  std::map<int, LmNet> get_net_map() { return _net_map; }
  LmNet* get_net(int net_id);
  int get_layer_order() { return _layer_order; }

  // setter
  void set_layer_name(std::string name) { _layer_name = name; }
  void set_as_routing(bool b_routing) { _b_routing = b_routing; }
  void set_layer_order(int order) { _layer_order = order; }
  void set_llx(int value) { _llx = value; }
  void set_lly(int value) { _lly = value; }
  void set_urx(int value) { _urx = value; }
  void set_ury(int value) { _ury = value; }

  // operator
  LmNet* getOrCreateNet(int net_id);

 private:
  std::string _layer_name;
  bool _b_routing;
  int _layer_order;
  int _llx;
  int _lly;
  int _urx;
  int _ury;
  int _row_num;  /// patch row number
  int _row_space;
  int _col_num;  /// patch col number
  int _col_space;
  LmLayerGrid _grid;
  std::vector<std::vector<LmPatch>> _patch_matrix;
  std::map<int, LmNet> _net_map;
};

class LmPatchLayers
{
 public:
  LmPatchLayers(){};
  ~LmPatchLayers() = default;

  // getter
  int get_layer_order_top() { return _layer_order_top; }
  int get_layer_order_bottom() { return _layer_order_bottom; }
  std::map<int, LmPatchLayer>& get_patch_layer_map() { return _patch_layers; }
  LmPatchLayer* findPatchLayer(int order);

  // setter
  void set_layer_order_top(int order) { _layer_order_top = order; }
  void set_layer_order_bottom(int order) { _layer_order_bottom = order; }

  // operator

 private:
  int _layer_order_top = -1;
  int _layer_order_bottom = -1;
  std::map<int, LmPatchLayer> _patch_layers;  /// int : layer order
};

}  // namespace ilm
