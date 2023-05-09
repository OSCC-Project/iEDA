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
 * @file		IdbFill.h
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

#include "../../../basic/geometry/IdbGeometry.h"

namespace idb {

class IdbLayer;
class IdbVia;

class IdbFillLayer
{
 public:
  IdbFillLayer();
  ~IdbFillLayer();

  // getter
  IdbLayer* get_layer() { return _layer; }
  std::vector<IdbRect*>& get_rect_list() { return _rect_list; }
  size_t get_rect_num() const { return _rect_list.size(); }
  IdbRect* get_rect(size_t index);

  // setter
  void set_layer(IdbLayer* layer) { _layer = layer; }
  void set_rect_list(std::vector<IdbRect*> rect_list) { _rect_list = rect_list; }
  IdbRect* add_rect();
  void add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void reset_rect();

  // operator

 private:
  IdbLayer* _layer;
  std::vector<IdbRect*> _rect_list;
};

class IdbFillVia
{
 public:
  IdbFillVia();
  ~IdbFillVia();

  // getter
  IdbVia* get_via() { return _via; }
  std::vector<IdbCoordinate<int32_t>*>& get_coordinate_list() { return _coordinate_list; }
  IdbCoordinate<int32_t>* get_coordinate(size_t index);

  // setter
  void set_via(IdbVia* via) { _via = via; }
  IdbCoordinate<int32_t>* add_coordinate(int32_t x, int32_t y);

  // operator

 private:
  IdbVia* _via;
  std::vector<IdbCoordinate<int32_t>*> _coordinate_list;
};

class IdbFill
{
 public:
  IdbFill();
  ~IdbFill();

  enum IdbFillType : uint8_t
  {
    kNone,
    kLayer,
    kVia,
    kMax
  };

  // getter
  const IdbFillType& get_type() const { return _type; }
  IdbFillLayer* get_layer() { return _layer; }
  IdbFillVia* get_via() { return _via; }

  // setter
  void set_layer(IdbFillLayer* layer) { _layer = layer; }
  void set_via(IdbFillVia* via) { _via = via; }
  void set_type(IdbFillType type) { _type = type; }
  void set_type_layer() { _type = IdbFillType::kLayer; }
  void set_type_via() { _type = IdbFillType::kVia; }

  // operator

 private:
  IdbFillType _type;
  IdbFillLayer* _layer;
  IdbFillVia* _via;
};

class IdbFillList
{
 public:
  IdbFillList();
  ~IdbFillList();

  // getter
  std::vector<IdbFill*>& get_fill_list() { return _fill_list; }
  const int32_t get_num_fill() const { return _num_fill; }
  //   IdbFill* find_fill_by_layer(string name);
  //   IdbFill* find_fill_by_via(string name);

  // setter
  IdbFillLayer* add_fill_layer(IdbLayer* layer = nullptr);
  IdbFillVia* add_fill_via(IdbVia* via = nullptr);

  void reset();

  // operator
  void init_fill_list(int32_t size) { _fill_list.reserve(size); }

 private:
  int32_t _num_fill;
  std::vector<IdbFill*> _fill_list;
};

}  // namespace idb
