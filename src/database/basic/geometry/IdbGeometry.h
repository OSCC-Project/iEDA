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
#ifndef IDB_DB_GEOMETRY
#define IDB_DB_GEOMETRY
#pragma once

/**
 * @project		iDB
 * @file		IdbGeometry.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe basic geometry data structure.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace idb {

template <typename T>
class IdbCoordinate
{
 public:
  IdbCoordinate() : _x(0), _y(0) {}
  IdbCoordinate(T x, T y) : _x(x), _y(y) {}
  ~IdbCoordinate() {}

  bool operator==(const IdbCoordinate& other) { return (_x == other._x && _y == other._y); }
  bool operator!=(const IdbCoordinate& other) { return !((*this) == other); }
  IdbCoordinate operator=(const IdbCoordinate& other)
  {
    _x = other._x;
    _y = other._y;
    return (*this);
  }

  // getter
  T& get_x() { return _x; }
  T& get_y() { return _y; }
  bool is_negative() { return _x < 0 || _y < 0 ? true : false; }

  bool is_init() { return (_x == 0 && _y == 0) ? false : true; }

  // setter
  void set_x(T x) { _x = x; }
  void set_y(T y) { _y = y; }
  void set_xy(T x, T y)
  {
    _x = x;
    _y = y;
  }

  // operation

 private:
  T _x;
  T _y;
};
enum class RectEdgePosition : int8_t
{
  kNone,
  kLeft,
  kRight,
  kBottom,
  kTop,
  kInside,
  kMax
};
class IdbRect
{
 public:
  IdbRect() : _lx(0), _ly(0), _hx(0), _hy(0){};
  IdbRect(const int32_t lx, const int32_t ly, const int32_t hx, const int32_t hy, int32_t width = -1);
  IdbRect(IdbCoordinate<int32_t>* coord_1, IdbCoordinate<int32_t>* coord_2, int32_t width)
      : IdbRect(coord_1->get_x(), coord_1->get_y(), coord_2->get_x(), coord_2->get_y(), width)
  {
  }
  IdbRect(IdbRect const& rect) : _lx(rect._lx), _ly(rect._ly), _hx(rect._hx), _hy(rect._hy) {}
  IdbRect(IdbRect* rect) : _lx(rect->_lx), _ly(rect->_ly), _hx(rect->_hx), _hy(rect->_hy) {}
  ~IdbRect() = default;

  // getter
  int32_t get_low_x() { return _lx; }
  int32_t get_low_y() { return _ly; }
  IdbCoordinate<int32_t> get_low_point() { return IdbCoordinate<int32_t>(_lx, _ly); }
  int32_t get_high_x() { return _hx; }
  int32_t get_high_y() { return _hy; }
  IdbCoordinate<int32_t> get_high_point() { return IdbCoordinate<int32_t>(_hx, _hy); }
  IdbCoordinate<int32_t> get_middle_point() { return IdbCoordinate<int32_t>((_lx + _hx) / 2, (_ly + _hy) / 2); }
  int32_t get_middle_point_x() { return (_lx + _hx) / 2; }
  int32_t get_middle_point_y() { return (_ly + _hy) / 2; }

  int32_t get_width() { return std::abs(_hx - _lx); }
  int32_t get_height() { return std::abs(_hy - _ly); }
  int32_t get_min_length() { return std::min(get_width(), get_height()); }
  uint64_t get_area() { return ((uint64_t) get_width()) * ((uint64_t) get_height()); }

  bool is_init() { return (_lx == 0 && _ly == 0 && _hx == 0 && _hy == 0) ? false : true; }

  // setter
  void set_low_x(const int32_t x) { _lx = x; }
  void set_low_y(const int32_t y) { _ly = y; }
  void set_high_x(const int32_t x) { _hx = x; }
  void set_high_y(const int32_t y) { _hy = y; }
  void set_rect(const int32_t lx, const int32_t ly, const int32_t hx, const int32_t hy)
  {
    _lx = lx;
    _ly = ly;
    _hx = hx;
    _hy = hy;
  };
  void set_rect(IdbRect* rect)
  {
    _lx = rect->_lx;
    _ly = rect->_ly;
    _hx = rect->_hx;
    _hy = rect->_hy;
  };

  // operator
  IdbRect& operator=(const IdbRect& other)
  {
    // IdbRect::operator=(other);
    _lx = other._lx;
    _ly = other._ly;
    _hx = other._hx;
    _hy = other._hy;
    return (*this);
  }

  IdbRect& moveByStep(const int32_t step_x, const int32_t step_y)
  {
    _lx += step_x;
    _ly += step_y;
    _hx += step_x;
    _hy += step_y;
    return (*this);
  }

  bool containPoint(IdbCoordinate<int32_t> point)
  {
    return point.get_x() >= _lx && point.get_x() <= _hx && point.get_y() >= _ly && point.get_y() <= _hy ? true : false;
  }

  bool containPoint(IdbCoordinate<int32_t>* point)
  {
    return point->get_x() >= _lx && point->get_x() <= _hx && point->get_y() >= _ly && point->get_y() <= _hy ? true : false;
  }

  void adjustCoordinate(IdbCoordinate<int32_t>* main_point, IdbCoordinate<int32_t>* follow_point, bool adjust_follow = false);

  RectEdgePosition findCoordinateEdgePosition(IdbCoordinate<int32_t> point)
  {
    if (containPoint(point)) {
      return RectEdgePosition::kInside;
    }

    if (point.get_x() < _lx) {
      return RectEdgePosition::kLeft;
    }

    if (point.get_x() > _hx) {
      return RectEdgePosition::kRight;
    }

    if (point.get_y() < _ly) {
      return RectEdgePosition::kBottom;
    }

    if (point.get_y() > _hy) {
      return RectEdgePosition::kTop;
    }

    return RectEdgePosition::kNone;
  }

  bool isIntersection(IdbRect rect);
  bool isIntersection(IdbRect* rect);

 private:
  int32_t _lx;
  int32_t _ly;
  int32_t _hx;
  int32_t _hy;
};

}  // namespace idb

#endif  // IDB_DB_GEOMETRY
