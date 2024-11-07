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
 * @file		patch.h
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

namespace ilm {

class LmPatchNode
{
 public:
  LmPatchNode() = default;
  ~LmPatchNode() = default;

  // getter

  // setter

  // operator

 private:
  int _x;
  int _y;
  int _row_id;  // node order of layer rows
  int _col_id;  // node order of layer cols
};

class LmPatch
{
 public:
  LmPatch() = default;
  ~LmPatch() = default;

  // getter
  std::vector<std::vector<LmPatchNode>>& get_node_matrix() { return _node_matrix; }
  LmPatchNode& get_node(int row_id, int col_id);

  // setter

  // operator

 private:
  int _llx;
  int _lly;
  int _urx;
  int _ury;
  int _row_id;  // patch order of layer rows
  int _col_id;  // patch order of layer cols

  std::vector<std::vector<LmPatchNode>> _node_matrix;
};

class LmPatchLayer
{
 public:
  LmPatchLayer() = default;
  ~LmPatchLayer() = default;

  // getter
  std::vector<std::vector<LmPatch>>& get_patch_matrix() { return _patch_matrix; }
  LmPatch& get_patch(int row_id, int col_id);

  // setter

  // operator

 private:
  std::string _layer_name = "";
  int _layer_order = -1;
  int _llx;
  int _lly;
  int _urx;
  int _ury;
  int _row_num;  /// patch row number
  int _row_space;
  int _col_num;  /// patch col number
  int _col_space;
  std::vector<std::vector<LmPatch>> _patch_matrix;
};

class LmPatchLayers
{
 public:
  LmPatchLayers(){};
  ~LmPatchLayers() = default;

  // getter
  std::map<int, LmPatchLayer>& get_patch_layers() { return _patch_layers; }
  LmPatchLayer* findPatchLayer(int order);

  // setter

  // operator

 private:
  int _layer_order_top = -1;
  int _layer_order_bottom = -1;
  std::map<int, LmPatchLayer> _patch_layers;  /// int : layer order
};

}  // namespace ilm
