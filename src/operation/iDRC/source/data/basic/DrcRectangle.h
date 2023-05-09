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
#ifndef IDRC_SRC_DB_DRC_RECTANGLE_H_
#define IDRC_SRC_DB_DRC_RECTANGLE_H_

#include "DrcCoordinate.h"

namespace idrc {

template <typename T>
class DrcRectangle
{
 public:
  DrcRectangle() {}
  DrcRectangle(int lb_x, int lb_y, int rt_x, int rt_y)
  {
    set_lb(lb_x, lb_y);
    set_rt(rt_x, rt_y);
  }
  DrcRectangle(const DrcRectangle& other) : _lb(other._lb), _rt(other._rt) {}
  DrcRectangle(DrcRectangle&& other) : _lb(std::move(other._lb)), _rt(std::move(other._rt)) {}
  ~DrcRectangle() {}
  DrcRectangle& operator=(const DrcRectangle& other)
  {
    _lb = other._lb;
    _rt = other._rt;
    return (*this);
  }
  DrcRectangle& operator=(DrcRectangle&& other)
  {
    _lb = std::move(other._lb);
    _rt = std::move(other._rt);
    return (*this);
  }
  bool operator==(const DrcRectangle& other) { return (_lb == other._lb && _rt == other._rt); }
  bool operator!=(const DrcRectangle& other) { return !((*this) == other); }
  // getter
  DrcCoordinate<T>& get_lb() { return _lb; }
  DrcCoordinate<T>& get_rt() { return _rt; }
  T get_lb_x() const { return _lb.get_x(); }
  T get_lb_y() const { return _lb.get_y(); }
  T get_rt_x() const { return _rt.get_x(); }
  T get_rt_y() const { return _rt.get_y(); }
  // setters
  void set_lb(const T x, const T y)
  {
    _lb.set_x(x);
    _lb.set_y(y);
  }
  void set_rt(const T x, const T y)
  {
    _rt.set_x(x);
    _rt.set_y(y);
  }

  // function

 private:
  DrcCoordinate<T> _lb;
  DrcCoordinate<T> _rt;
};

}  // namespace idrc
#endif  // IDRC_SRC_DB_RECTANGLET_H_
