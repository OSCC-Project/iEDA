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
  PlanarRect(const irt_int lb_x, const irt_int lb_y, const irt_int rt_x, const irt_int rt_y)
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
  irt_int get_lb_x() const { return _lb.get_x(); }
  irt_int get_lb_y() const { return _lb.get_y(); }
  irt_int get_rt_x() const { return _rt.get_x(); }
  irt_int get_rt_y() const { return _rt.get_y(); }
  // const getter
  const PlanarCoord& get_lb() const { return _lb; }
  const PlanarCoord& get_rt() const { return _rt; }
  // setter
  void set_lb(const PlanarCoord& lb) { _lb = lb; }
  void set_rt(const PlanarCoord& rt) { _rt = rt; }
  void set_lb(const irt_int x, const irt_int y) { _lb.set_coord(x, y); }
  void set_rt(const irt_int x, const irt_int y) { _rt.set_coord(x, y); }
  void set_lb_x(const irt_int lb_x) { _lb.set_x(lb_x); }
  void set_lb_y(const irt_int lb_y) { _lb.set_y(lb_y); }
  void set_rt_x(const irt_int rt_x) { _rt.set_x(rt_x); }
  void set_rt_y(const irt_int rt_y) { _rt.set_y(rt_y); }
  // function
  inline irt_int getXSpan() const;
  inline irt_int getYSpan() const;
  inline irt_int getLength() const;
  inline irt_int getWidth() const;
  inline Direction getRectDirection(Direction point_direction) const;
  inline irt_int getHalfPerimeter() const;
  inline irt_int getPerimeter() const;
  inline double getArea() const;
  inline std::vector<Segment<PlanarCoord>> getEdgeList() const;
  inline PlanarCoord getMidPoint() const;

 private:
  PlanarCoord _lb;
  PlanarCoord _rt;
};

inline irt_int PlanarRect::getXSpan() const
{
  return get_rt_x() - get_lb_x();
}

inline irt_int PlanarRect::getYSpan() const
{
  return get_rt_y() - get_lb_y();
}

inline irt_int PlanarRect::getLength() const
{
  return std::max(getXSpan(), getYSpan());
}

inline irt_int PlanarRect::getWidth() const
{
  return std::min(getXSpan(), getYSpan());
}

inline Direction PlanarRect::getRectDirection(Direction point_direction = Direction::kNone) const
{
  Direction direction = Direction::kNone;

  irt_int x_length = get_rt_x() - get_lb_x();
  irt_int y_length = get_rt_y() - get_lb_y();

  if (x_length > y_length) {
    direction = Direction::kHorizontal;
  } else if (x_length < y_length) {
    direction = Direction::kVertical;
  } else {
    direction = point_direction;
  }

  return direction;
}

inline irt_int PlanarRect::getHalfPerimeter() const
{
  return getLength() + getWidth();
}

inline irt_int PlanarRect::getPerimeter() const
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
  irt_int lb_x = _lb.get_x();
  irt_int lb_y = _lb.get_y();
  irt_int rt_x = _rt.get_x();
  irt_int rt_y = _rt.get_y();
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
