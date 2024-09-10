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
 * @file		IdbViaMaster.h
 * @date		25/05/2021
 * @version		0.1
 * @description

describe via master in deflefref P.137
A fixed via is defined using rectangles or polygons, and does not use a VIARULE. The fixed
via name must mean the same via in all associated LEF and DEF files.
A generated via is defined using VIARULE parameters to indicate that it was derived from a
VIARULE GENERATE statement.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../../../basic/geometry/IdbLayerShape.h"
#include "IdbLayer.h"
#include "IdbViaRule.h"

namespace idb {

using std::pair;
using std::string;
using std::vector;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
{ VIARULE viaRuleName ;
  CUTSIZE xSize ySize ;
  LAYERS botMetalLayer cutLayer topMetalLayer ;
  CUTSPACING xCutSpacing yCutSpacing ;
  ENCLOSURE xBotEnc yBotEnc xTopEnc yTopEnc ;
  [ROWCOL numCutRows numCutCols ;]
  [ORIGIN xOffset yOffset ;]
  [OFFSET xBotOffset yBotOffset xTopOffset yTopOffset ;]
  [PATTERN cutPattern ;]
}
*/

class IdbLayerRouting;
class IdbLayerCut;
class IdbViaRuleGenerate;

class IdbViaMasterRulePattern
{
 public:
  IdbViaMasterRulePattern() = default;
  ~IdbViaMasterRulePattern() = default;

  /// getter
  const string get_pattern_string() { return _pattern; }
  bool is_cut_exist(size_t row, size_t col)
  {
    if (row < _row_num && col < _col_num) {
      return (bool) _pattern_state[row][col];
    }

    return false;
  }

  /// setter
  void set_pattern(string pattern, size_t row, size_t col)
  {
    _pattern = pattern;
    parse_pattern(row, col);
  }
  bool set_pattern_value(size_t row, size_t col, int8_t value)
  {
    if (row < _row_num && col < _col_num) {
      _pattern_state[row][col] = value;
      return true;
    }
    return false;
  }

 private:
  string _pattern;

  /// definition
  constexpr static char repeat_flag = 'R';
  constexpr static size_t repeat_char_len = 3;
  constexpr static size_t repeat_unit_len = 4;

  /// using these data to calculate pattern string
  vector<vector<int8_t>> _pattern_state;
  size_t _row_num = 0;
  size_t _col_num = 0;

  /// operator
  string build_pattern();
  void parse_pattern(size_t row, size_t col);
  void parse_pattern_array(vector<pair<string, string>>& pattern_array);
  void parse_pattern_row_value(size_t row_index, string value);
  bool save_pattern_value(char value, int row_index, int& col_index);

  int hexString2Int(string hex_string);
  int8_t getBitVaule(int8_t& value, int32_t index);
};

class IdbViaMasterGenerate
{
 public:
  IdbViaMasterGenerate();
  ~IdbViaMasterGenerate();

  // getter
  const string get_rule_name() const { return _rule_name; }
  IdbViaRuleGenerate* get_rule_generate() { return _rule_generate; }
  const int32_t get_cut_size_x() const { return _cut_size_x; }
  const int32_t get_cut_size_y() const { return _cut_size_y; }
  IdbLayerRouting* get_layer_bottom() { return _layer_bottom; }
  IdbLayerCut* get_layer_cut() { return _layer_cut; }
  IdbLayerRouting* get_layer_top() { return _layer_top; }
  int32_t get_cut_spcing_x() { return _cut_spacing_x; }
  int32_t get_cut_spcing_y() { return _cut_spacing_y; }
  const int32_t get_enclosure_bottom_x() const { return _enclosure_bottom_x; }
  const int32_t get_enclosure_bottom_y() const { return _enclosure_bottom_y; }
  const int32_t get_enclosure_top_x() const { return _enclosure_top_x; }
  const int32_t get_enclosure_top_y() const { return _enclosure_top_y; }
  // tbd*****
  const int32_t get_cut_rows() const { return _num_cut_rows; }
  const int32_t get_cut_cols() const { return _num_cut_cols; }
  const int32_t get_original_offset_x() const { return _original_offset_x; }
  const int32_t get_original_offset_y() const { return _original_offset_y; }
  const int32_t get_offset_bottom_x() const { return _offset_bottom_x; }
  const int32_t get_offset_bottom_y() const { return _offset_bottom_y; }
  const int32_t get_offset_top_x() const { return _offset_top_x; }
  const int32_t get_offset_top_y() const { return _offset_top_y; }
  vector<IdbRect*>& get_cut_rect_list() { return _cut_rect_list; }
  IdbRect* get_cut_bouding_rect() { return _cut_bouding_rect; }
  IdbViaMasterRulePattern* get_patttern() { return _patttern; }
  bool is_pattern_cut_exist(int32_t row, int32_t col) { return _patttern == nullptr ? false : _patttern->is_cut_exist(row, col); }

  // setter
  void set_rule_name(string name) { _rule_name = name; }
  void set_rule_generate(IdbViaRuleGenerate* rule) { _rule_generate = rule; }
  void set_cut_size(int32_t x, int32_t y)
  {
    _cut_size_x = x;
    _cut_size_y = y;
  }
  void set_layer_bottom(IdbLayerRouting* layer_bottom) { _layer_bottom = layer_bottom; }
  void set_layer_cut(IdbLayerCut* layer_cut) { _layer_cut = layer_cut; }
  void set_layer_top(IdbLayerRouting* layer_top) { _layer_top = layer_top; }

  void set_cut_spacing(int32_t x, int32_t y)
  {
    _cut_spacing_x = x;
    _cut_spacing_y = y;
  }
  void set_enclosure_bottom(int32_t bottom_x, int32_t bottom_y)
  {
    _enclosure_bottom_x = bottom_x;
    _enclosure_bottom_y = bottom_y;
  }
  void set_enclosure_top(int32_t top_x, int32_t top_y)
  {
    _enclosure_top_x = top_x;
    _enclosure_top_y = top_y;
  }
  void set_cut_row_col(int32_t rows, int32_t cols)
  {
    _num_cut_rows = rows;
    _num_cut_cols = cols;
  }
  void set_original(int32_t offset_x, int32_t offset_y)
  {
    _original_offset_x = offset_x;
    _original_offset_y = offset_y;
  }
  void set_offset_bottom(int32_t x, int32_t y)
  {
    _offset_bottom_x = x;
    _offset_bottom_y = y;
  }
  void set_offset_top(int32_t x, int32_t y)
  {
    _offset_top_x = x;
    _offset_top_y = y;
  }
  void set_patttern(IdbViaMasterRulePattern* patttern) { _patttern = patttern; }
  void set_patttern(string pattern_string)
  {
    if (pattern_string.empty()) {
      return;
    }

    if (nullptr == _patttern) {
      _patttern = new IdbViaMasterRulePattern();
    }
    _patttern->set_pattern(pattern_string, _num_cut_rows, _num_cut_cols);
  }

  IdbRect* add_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void set_cut_bouding_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y) { _cut_bouding_rect->set_rect(ll_x, ll_y, ur_x, ur_y); }
  // operator
  IdbViaMasterGenerate* clone();
  void clear();

 private:
  string _rule_name;
  IdbViaRuleGenerate* _rule_generate;
  int32_t _cut_size_x;
  int32_t _cut_size_y;
  IdbLayerRouting* _layer_bottom;
  IdbLayerCut* _layer_cut;
  IdbLayerRouting* _layer_top;
  int32_t _cut_spacing_x;
  int32_t _cut_spacing_y;
  int32_t _enclosure_bottom_x;
  int32_t _enclosure_bottom_y;
  int32_t _enclosure_top_x;
  int32_t _enclosure_top_y;

  int32_t _num_cut_rows;
  int32_t _num_cut_cols;

  int32_t _original_offset_x;
  int32_t _original_offset_y;

  int32_t _offset_bottom_x;
  int32_t _offset_bottom_y;
  int32_t _offset_top_x;
  int32_t _offset_top_y;
  //<-----tbd----------------
  // pattern
  IdbViaMasterRulePattern* _patttern;

  vector<IdbRect*> _cut_rect_list;  // rect list of each cut
  IdbRect* _cut_bouding_rect;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
{[RESISTANCE resistValue ;]
  {LAYER layerName ;
    { RECT [MASK maskNum] pt pt ;
    | POLYGON [MASK maskNum] pt pt pt ...;} ...
  }...
}
*/

class IdbViaMasterFixed
{
 public:
  IdbViaMasterFixed();
  ~IdbViaMasterFixed();

  // getter
  IdbLayerShape* get_layer_shape() { return _layer_shape; }
  IdbLayer* get_layer() { return _layer_shape->get_layer(); }
  vector<IdbRect*>& get_rect_list() { return _layer_shape->get_rect_list(); }
  IdbRect* get_rect(int32_t index);

  // setter
  IdbLayerShape* set_layer_shape(IdbLayerShape* layer_shape)
  {
    _layer_shape = layer_shape;
    return _layer_shape;
  }
  void set_layer(IdbLayer* layer) { _layer_shape->set_layer(layer); }
  IdbRect* add_rect();
  void add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);

  // operator
  void clear();

 private:
  IdbLayerShape* _layer_shape;
  //   IdbLayer* _layer;
  //   vector<IdbRect*> _rect_list;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbViaMaster
{
 public:
  IdbViaMaster();
  ~IdbViaMaster();

  enum class IdbViaMasterType
  {
    kNone,
    kViaRule,
    kFixed,
    kMax
  };
  enum IdbViaLayerIndex : int8_t
  {
    kNone = 0,
    kLayerBottom = 1,
    kLayerCut = 2,
    kLayerTop = 3,
    kMax = 3
  };

  // getter
  const string get_name() const { return _name; }
  const bool is_default() const { return _is_default; }
  IdbViaMasterType get_type() const { return _type; }
  bool is_fix() { return _type == IdbViaMasterType::kFixed ? true : false; }
  bool is_generate() { return _type == IdbViaMasterType::kViaRule ? true : false; }
  IdbViaMasterGenerate* get_master_generate() { return _master_generate; }
  vector<IdbViaMasterFixed*>& get_master_fixed_list() { return _master_fixed_list; }
  IdbViaMasterFixed* get_master_fixed(IdbViaLayerIndex layer_index);
  IdbRect* get_cut_rect() { return _cut_rect; }

  IdbLayerShape* get_bottom_layer_shape() { return _layer_shape_bottom; }
  IdbLayerShape* get_top_layer_shape() { return _layer_shape_top; }
  IdbLayerShape* get_cut_layer_shape() { return _layer_shape_cut; }

  const int32_t get_cut_rows() const { return _num_cut_rows; }
  const int32_t get_cut_cols() const { return _num_cut_cols; }
  bool isOneCut();

  // setter
  void set_name(string name) { _name = name; }
  void set_default(bool is_default) { _is_default = is_default; }
  void set_type(IdbViaMasterType type) { _type = type; }
  void set_type_fixed() { _type = IdbViaMasterType::kFixed; }
  void set_type_generate() { _type = IdbViaMasterType::kViaRule; }
  void set_master_generate(IdbViaMasterGenerate* master_generate) { _master_generate = master_generate; }
  void set_master_fixed_list(vector<IdbViaMasterFixed*> fixed_list) { _master_fixed_list = fixed_list; }
  IdbViaMasterFixed* add_fixed(string layer_name);

  void set_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);

  void set_via_shape()
  {
    set_bottom_layer_shape();
    set_top_layer_shape();
    set_cut_layer_shape();
  };
  void set_bottom_layer_shape();
  void set_top_layer_shape();
  void set_cut_layer_shape();
  void set_layer_shape(IdbViaLayerIndex layer_index);

  void set_cut_row_col(int32_t rows, int32_t cols)
  {
    _num_cut_rows = rows;
    _num_cut_cols = cols;
  }

  // operator
  IdbViaMaster* clone();
  void clear();

 private:
  string _name;
  IdbRect* _cut_rect;  // the core area of cut
  IdbViaMasterGenerate* _master_generate;
  bool _is_default;
  IdbViaMasterType _type;
  int32_t _num_cut_rows = -1;
  int32_t _num_cut_cols = -1;
  vector<IdbViaMasterFixed*> _master_fixed_list;

  IdbLayerShape* _layer_shape_bottom;
  IdbLayerShape* _layer_shape_cut;
  IdbLayerShape* _layer_shape_top;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbViaMasterList
{
 public:
  IdbViaMasterList();
  ~IdbViaMasterList();

  // getter
  vector<IdbViaMaster*>& get_master_list() { return _via_master_list; }
  int32_t get_num_via_master() { return _num_master; }
  IdbViaMaster* find_via_master(string name);
  IdbViaMaster* find_via_master(int32_t index);

  // setter
  void set_num_via_master(uint32_t number) { _num_master = number; }
  IdbViaMaster* add_via_master(IdbViaMaster* via_master = nullptr);
  IdbViaMaster* add_via_master(string name);

  void reset();

  // operator
  void init_via_master_list(int32_t size) { _via_master_list.reserve(size); }

 private:
  int32_t _num_master;
  vector<IdbViaMaster*> _via_master_list;
};

}  // namespace idb
