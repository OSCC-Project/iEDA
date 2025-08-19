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
  PlanarRect(const PlanarCoord& ll, const PlanarCoord& ur)
  {
    _ll = ll;
    _ur = ur;
  }
  PlanarRect(const int32_t ll_x, const int32_t ll_y, const int32_t ur_x, const int32_t ur_y)
  {
    set_ll(ll_x, ll_y);
    set_ur(ur_x, ur_y);
  }
  ~PlanarRect() = default;
  bool operator==(const PlanarRect& other) const { return (_ll == other._ll && _ur == other._ur); }
  bool operator!=(const PlanarRect& other) const { return !((*this) == other); }
  // getter
  PlanarCoord& get_ll() { return _ll; }
  PlanarCoord& get_ur() { return _ur; }
  int32_t get_ll_x() const { return _ll.get_x(); }
  int32_t get_ll_y() const { return _ll.get_y(); }
  int32_t get_ur_x() const { return _ur.get_x(); }
  int32_t get_ur_y() const { return _ur.get_y(); }
  // const getter
  const PlanarCoord& get_ll() const { return _ll; }
  const PlanarCoord& get_ur() const { return _ur; }
  // setter
  void set_ll(const PlanarCoord& ll) { _ll = ll; }
  void set_ur(const PlanarCoord& ur) { _ur = ur; }
  void set_ll(const int32_t x, const int32_t y) { _ll.set_coord(x, y); }
  void set_ur(const int32_t x, const int32_t y) { _ur.set_coord(x, y); }
  void set_ll_x(const int32_t ll_x) { _ll.set_x(ll_x); }
  void set_ll_y(const int32_t ll_y) { _ll.set_y(ll_y); }
  void set_ur_x(const int32_t ur_x) { _ur.set_x(ur_x); }
  void set_ur_y(const int32_t ur_y) { _ur.set_y(ur_y); }
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
  inline bool isIncorrect() const;

 private:
  PlanarCoord _ll;
  PlanarCoord _ur;
};

inline int32_t PlanarRect::getXSpan() const
{
  return get_ur_x() - get_ll_x();
}

inline int32_t PlanarRect::getYSpan() const
{
  return get_ur_y() - get_ll_y();
}

inline int32_t PlanarRect::getLength() const
{
  return std::max(getXSpan(), getYSpan());
}

inline int32_t PlanarRect::getWidth() const
{
  return std::min(getXSpan(), getYSpan());
}

inline Direction PlanarRect::getRectDirection(Direction point_direction) const
{
  Direction direction = Direction::kNone;

  int32_t x_length = get_ur_x() - get_ll_x();
  int32_t y_length = get_ur_y() - get_ll_y();

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
  int32_t ll_x = _ll.get_x();
  int32_t ll_y = _ll.get_y();
  int32_t ur_x = _ur.get_x();
  int32_t ur_y = _ur.get_y();
  std::vector<Segment<PlanarCoord>> segment_list;
  if (ll_x == ur_x || ll_y == ur_y) {
    segment_list.emplace_back(_ll, _ur);
  } else {
    segment_list.emplace_back(_ll, PlanarCoord(ll_x, ur_y));
    segment_list.emplace_back(_ll, PlanarCoord(ur_x, ll_y));
    segment_list.emplace_back(PlanarCoord(ll_x, ur_y), _ur);
    segment_list.emplace_back(PlanarCoord(ur_x, ll_y), _ur);
  }
  return segment_list;
}

inline PlanarCoord PlanarRect::getMidPoint() const
{
  return PlanarCoord((get_ll_x() + get_ur_x()) / 2, (get_ll_y() + get_ur_y()) / 2);
}

inline bool PlanarRect::isIncorrect() const
{
  if (_ll.get_x() > _ur.get_x() || _ll.get_y() > _ur.get_y()) {
    return true;
  } else {
    return false;
  }
}

struct CmpPlanarRectByXASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_ll() == b.get_ll()) {
      return CmpPlanarCoordByXASC()(a.get_ur(), b.get_ur());
    } else {
      return CmpPlanarCoordByXASC()(a.get_ll(), b.get_ll());
    }
  }
};

struct CmpPlanarRectByYASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_ll() == b.get_ll()) {
      return CmpPlanarCoordByYASC()(a.get_ur(), b.get_ur());
    } else {
      return CmpPlanarCoordByYASC()(a.get_ll(), b.get_ll());
    }
  }
};

}  // namespace irt
