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
 * @project		iDB
 * @file		IdbVias.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Vias information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

// #include "IdbViaMaster.h"
#include "../../../basic/geometry/IdbGeometry.h"
#include "../../../basic/geometry/IdbLayerShape.h"
#include "../IdbEnum.h"
#include "../IdbObject.h"

namespace idb {

using std::string;
using std::vector;

class IdbViaMaster;
class IdbLayerCut;
class IdbLayerCutArraySpacing;

class IdbVia : public IdbObject
{
 public:
  IdbVia();
  ~IdbVia();

  // getter
  const string& get_name() const { return _name; }
  IdbViaMaster* get_instance();
  IdbCoordinate<int32_t>* get_coordinate() { return _coordinate; }

  IdbLayerShape get_bottom_layer_shape();
  IdbRect get_bottom_bounding_box();
  IdbLayerShape get_top_layer_shape();
  IdbRect get_top_bounding_box();
  IdbLayerShape get_cut_layer_shape();
  IdbRect get_cut_bounding_box();

  // setter
  void set_name(string name) { _name = name; }
  void set_instance(IdbViaMaster* instance);
  void reset_instance(IdbViaMaster* instance);
  void set_coordinate(IdbCoordinate<int32_t>* point);
  void set_coordinate(int32_t x, int32_t y) { _coordinate->set_xy(x, y); }

  // operator
  IdbVia* clone();
  void clear();

  bool isIntersection(IdbRect rect, IdbLayer* layer);

 private:
  string _name;
  IdbViaMaster* _master_instance;
  bool _b_master_clone = false;
  IdbCoordinate<int32_t>* _coordinate;
};

class IdbVias
{
 public:
  IdbVias();
  ~IdbVias();

  // getter
  vector<IdbVia*>& get_via_list() { return _via_list; }
  const size_t get_num_via() const { return _num_vias; }
  IdbVia* find_via(const string& name);
  IdbVia* find_via(size_t index);
  IdbVia* find_via_generate(IdbLayerCut* layer_cut, int32_t width = 0, int32_t height = 0, string via_name = "");

  // setter

  void set_num_via(size_t number) { _num_vias = number; }
  IdbVia* add_via(IdbVia* via = nullptr);
  IdbVia* add_via(string name);

  void reset();
  void init_via_list(int32_t size) { _via_list.reserve(size); }

  // operator
  IdbVia* createVia(string via_name, IdbLayerCut* layer_cut, int32_t width_design = 0, int32_t height_design = 0,
                    IdbLayerDirection direction = IdbLayerDirection::kNone);
  //   IdbVia* createViaDefault(string via_name, IdbLayerCut* layer_cut);
  std::pair<int32_t, int32_t> calculateRowsCols(IdbLayerCut* layer_cut, int32_t width = 0, int32_t height = 0);
  string createViaPatternString(int row_num, int col_num, IdbLayerCutArraySpacing* _array_spacing);
  string constructRowPattern(int col_num, int array_cut_num);
  string transIntToHexBy4Bits(uint8_t value);
  string transInt8ToHex(uint8_t value);

 private:
  constexpr static size_t kMinRowColNum = 1;
  size_t _num_vias;
  vector<IdbVia*> _via_list;
};

}  // namespace idb
