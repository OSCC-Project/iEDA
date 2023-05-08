#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		IdbVias.h
 * @copyright	(c) 2021 All Rights Reserved.
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
#include "../IdbObject.h"
// #include "../db_layout/IdbLayer.h"

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
  IdbViaMaster* get_instance() { return _master_instance; }
  IdbCoordinate<int32_t>* get_coordinate() { return _coordinate; }

  IdbLayerShape get_bottom_layer_shape();
  IdbRect get_bottom_bounding_box();
  IdbLayerShape get_top_layer_shape();
  IdbRect get_top_bounding_box();
  IdbLayerShape get_cut_layer_shape();
  IdbRect get_cut_bounding_box();

  // setter
  void set_name(string name) { _name = name; }
  void set_instance(IdbViaMaster* instance) { _master_instance = instance; }
  void set_coordinate(IdbCoordinate<int32_t>* point);
  void set_coordinate(int32_t x, int32_t y) { _coordinate->set_xy(x, y); }
  //   bool set_bounding_box();

  // operator
  IdbVia* clone();
  void clear();

  bool isIntersection(IdbRect rect, IdbLayer* layer);

 private:
  string _name;
  IdbViaMaster* _master_instance;
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
  IdbVia* createVia(string via_name, IdbLayerCut* layer_cut, int32_t width_design = 0, int32_t height_design = 0);
  IdbVia* createViaDefault(string via_name, IdbLayerCut* layer_cut);
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
