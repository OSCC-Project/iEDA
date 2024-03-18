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

#include "Direction.hpp"
#include "PlanarCoord.hpp"
#include "Segment.hpp"

namespace irt {

class PlanarRect
{
 public:
  PlanarRect() = default;
  PlanarRect(const PlanarCoord& lb, const PlanarCoord& rt)
  {
    _lb = lb;
    _rt = rt;
  }
  PlanarRect(const int32_t lb_x, const int32_t lb_y, const int32_t rt_x, const int32_t rt_y)
  {
    set_lb(lb_x, lb_y);
    set_rt(rt_x, rt_y);
  }
  ~PlanarRect() = default;
  bool operator==(const PlanarRect& other) const { return (_lb == other._lb && _rt == other._rt); }
  bool operator!=(const PlanarRect& other) const { return !((*this) == other); }
  // getter
  PlanarCoord& get_lb() { return _lb; }
  PlanarCoord& get_rt() { return _rt; }
  int32_t get_lb_x() const { return _lb.get_x(); }
  int32_t get_lb_y() const { return _lb.get_y(); }
  int32_t get_rt_x() const { return _rt.get_x(); }
  int32_t get_rt_y() const { return _rt.get_y(); }
  // const getter
  const PlanarCoord& get_lb() const { return _lb; }
  const PlanarCoord& get_rt() const { return _rt; }
  // setter
  void set_lb(const PlanarCoord& lb) { _lb = lb; }
  void set_rt(const PlanarCoord& rt) { _rt = rt; }
  void set_lb(const int32_t x, const int32_t y) { _lb.set_coord(x, y); }
  void set_rt(const int32_t x, const int32_t y) { _rt.set_coord(x, y); }
  void set_lb_x(const int32_t lb_x) { _lb.set_x(lb_x); }
  void set_lb_y(const int32_t lb_y) { _lb.set_y(lb_y); }
  void set_rt_x(const int32_t rt_x) { _rt.set_x(rt_x); }
  void set_rt_y(const int32_t rt_y) { _rt.set_y(rt_y); }
  // function
  inline int32_t getXSpan() const;
  inline int32_t getYSpan() const;
  inline int32_t getLength() const;
  inline int32_t getWidth() const;
  inline Direction getRectDirection(Direction point_direction) const;
  inline int32_t getHalfPerimeter() const;
  inline int32_t getPerimeter() const;
  inline double getArea() const;
  inline std::vector<Segment<PlanarCoord>> getEdgeList() const;
  inline PlanarCoord getMidPoint() const;
  inline bool isIncorrected() const;

 private:
  PlanarCoord _lb;
  PlanarCoord _rt;
};

inline int32_t PlanarRect::getXSpan() const
{
  return get_rt_x() - get_lb_x();
}

inline int32_t PlanarRect::getYSpan() const
{
  return get_rt_y() - get_lb_y();
}

inline int32_t PlanarRect::getLength() const
{
  return std::max(getXSpan(), getYSpan());
}

inline int32_t PlanarRect::getWidth() const
{
  return std::min(getXSpan(), getYSpan());
}

inline Direction PlanarRect::getRectDirection(Direction point_direction = Direction::kNone) const
{
  Direction direction = Direction::kNone;

  int32_t x_length = get_rt_x() - get_lb_x();
  int32_t y_length = get_rt_y() - get_lb_y();

  if (x_length > y_length) {
    direction = Direction::kHorizontal;
  } else if (x_length < y_length) {
    direction = Direction::kVertical;
  } else {
    direction = point_direction;
  }

  return direction;
}

inline int32_t PlanarRect::getHalfPerimeter() const
{
  return getLength() + getWidth();
}

inline int32_t PlanarRect::getPerimeter() const
{
  return getHalfPerimeter() * 2;
}

inline double PlanarRect::getArea() const
{
  double area = getLength();
  area *= getWidth();
  return area;
}

inline std::vector<Segment<PlanarCoord>> PlanarRect::getEdgeList() const
{
  int32_t lb_x = _lb.get_x();
  int32_t lb_y = _lb.get_y();
  int32_t rt_x = _rt.get_x();
  int32_t rt_y = _rt.get_y();
  std::vector<Segment<PlanarCoord>> segment_list;
  if (lb_x == rt_x || lb_y == rt_y) {
    segment_list.emplace_back(_lb, _rt);
  } else {
    segment_list.emplace_back(_lb, PlanarCoord(lb_x, rt_y));
    segment_list.emplace_back(_lb, PlanarCoord(rt_x, lb_y));
    segment_list.emplace_back(PlanarCoord(lb_x, rt_y), _rt);
    segment_list.emplace_back(PlanarCoord(rt_x, lb_y), _rt);
  }
  return segment_list;
}

inline PlanarCoord PlanarRect::getMidPoint() const
{
  return PlanarCoord((get_lb_x() + get_rt_x()) / 2, (get_lb_y() + get_rt_y()) / 2);
}

inline bool PlanarRect::isIncorrected() const
{
  if (_lb.get_x() > _rt.get_x() || _lb.get_y() > _rt.get_y()) {
    return true;
  } else {
    return false;
  }
}

struct CmpPlanarRectByXASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpPlanarCoordByXASC()(a.get_rt(), b.get_rt());
    } else {
      return CmpPlanarCoordByXASC()(a.get_lb(), b.get_lb());
    }
  }
};

struct CmpPlanarRectByYASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpPlanarCoordByYASC()(a.get_rt(), b.get_rt());
    } else {
      return CmpPlanarCoordByYASC()(a.get_lb(), b.get_lb());
    }
  }
};

}  // namespace irt
