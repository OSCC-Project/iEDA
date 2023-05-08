#ifndef IDB_DB_LAYER_SHAPE
#define IDB_DB_LAYER_SHAPE
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
 * @file		IdbLayerShape.h
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


        #Describe layer geometry shape.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

 private:
  IdbLayerShapeType _type;
  IdbLayer* _layer;
  std::vector<IdbRect*> _rect_list;
};
}  // namespace idb

#endif  // IDB_DB_LAYER_SHAPE
