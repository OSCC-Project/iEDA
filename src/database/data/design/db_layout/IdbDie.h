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
 * @file		IdbDie.h
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Die Area information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../IdbObject.h"
// #include "../../../basic/geometry/IdbGeometry.h"

namespace idb {

using std::vector;

class IdbDie : public IdbObject
{
 public:
  IdbDie();
  ~IdbDie();

  // getter
  vector<IdbCoordinate<int32_t>*>& get_points() { return _points; }
  const float get_utilization() const { return _utilization; }
  const uint64_t get_area() const { return ((uint64_t) _width) * ((uint64_t) _height); }
  const int32_t get_llx() { return std::min(_points[0]->get_x(), _points[1]->get_x()); }
  const int32_t get_lly() { return std::min(_points[0]->get_y(), _points[1]->get_y()); }
  const int32_t get_urx() { return std::max(_points[0]->get_x(), _points[1]->get_x()); }
  const int32_t get_ury() { return std::max(_points[0]->get_y(), _points[1]->get_y()); }
  const int32_t get_width() { return _width; }
  const int32_t get_height() { return _height; }

  // setter
  void set_points(vector<IdbCoordinate<int32_t>*> points);
  void set_width(int32_t width) { _width = width; }
  void set_height(int32_t height) { _height = height; }
  void set_utilization(float utilization) { _utilization = utilization; }
  bool set_bounding_box();

  // operator
  void reset()
  {
    for (auto pt : _points) {
      if (nullptr != pt) {
        delete pt;
        pt = nullptr;
      }
    }

    _points.clear();
  }

  uint32_t add_point(IdbCoordinate<int32_t>* pt);
  uint32_t add_point(int32_t x, int32_t y);

  // verify data
  void print();

 private:
  constexpr static size_t kMaxPointsNumber = 2;
  /// @brief only support rectangle
  vector<IdbCoordinate<int32_t>*> _points;
  int32_t _width;
  int32_t _height;

  float _utilization;
};

}  // namespace idb
