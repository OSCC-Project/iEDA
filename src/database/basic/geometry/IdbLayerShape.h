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
 * @file		IdbLayerShape.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        #Describe layer geometry shape.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "IdbGeometry.h"

namespace idb {

enum class IdbLayerShapeType
{
  kNone,
  kRect,
  kPath,
  kVia,
  kMax
};

class IdbLayer;
class IdbVia;

class IdbLayerShape
{
 public:
  IdbLayerShape(IdbLayerShapeType type = IdbLayerShapeType::kRect);
  IdbLayerShape(const IdbLayerShape& other);
  IdbLayerShape(IdbLayerShape&& other);

  ~IdbLayerShape();

  // getter
  IdbLayerShapeType& get_type() { return _type; }
  bool is_rect() { return _type == IdbLayerShapeType::kRect ? true : false; }
  bool is_path() { return _type == IdbLayerShapeType::kPath ? true : false; }
  bool is_via() { return _type == IdbLayerShapeType::kVia ? true : false; }
  IdbRect* get_rect(uint index);
  uint32_t get_rect_list_num() { return _rect_list.size(); }
  std::vector<IdbRect*>& get_rect_list() { return _rect_list; }
  IdbCoordinate<int32_t> get_average_xy();
  IdbLayer* get_layer() { return _layer; }
  IdbRect get_bounding_box();

  // setter
  void set_type_rect() { _type = IdbLayerShapeType::kRect; }
  void set_type_via() { _type = IdbLayerShapeType::kVia; }
  void set_type_path() { _type = IdbLayerShapeType::kPath; }
  void add_rect(IdbRect* rect);
  void add_rect(IdbRect rect);
  void add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void set_layer(IdbLayer* layer) { _layer = layer; }

  // opereator
  IdbLayerShape& operator=(const IdbLayerShape& other);
  void moveToLocation(IdbCoordinate<int32_t>* coordinate);
  void clone(IdbLayerShape& layer_shape);
  void clear();
  IdbRect* contains(IdbCoordinate<int32_t>* coordinate, IdbLayer* layer)
  {
    if (_layer != layer || coordinate == nullptr) {
      return nullptr;
    }

    for (auto& rect : _rect_list) {
      if (nullptr != rect && rect->containPoint(coordinate)) {
        return rect;
      }
    }

    return nullptr;
  }

  bool isIntersected(int x, int y, IdbLayer* layer = nullptr);
  bool isIntersected(int llx, int lly, int urx, int ury, IdbLayer* layer = nullptr);
  bool isIntersected(IdbRect* rect_check, IdbLayer* layer = nullptr);

 private:
  IdbLayerShapeType _type;
  IdbLayer* _layer;
  std::vector<IdbRect*> _rect_list;
};
}  // namespace idb