/*
 * @Author: S.J Chen
 * @Date: 2022-01-20 19:28:55
 * @LastEditTime: 2022-01-20 19:48:45
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/ipl-shape/Point.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_POINT_H
#define IPL_POINT_H

namespace ipl {

template <class T>
class Point
{
 public:
  Point() : _x(-1), _y(-1) {}
  Point(T x, T y) : _x(x), _y(y) {}
  Point(const Point& other)
  {
    _x = other._x;
    _y = other._y;
  }
  Point(Point&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
  }
  ~Point() = default;

  Point& operator=(const Point& other)
  {
    _x = other._x;
    _y = other._y;
    return (*this);
  }

  Point& operator=(Point&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
    return (*this);
  }

  bool operator==(const Point& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const Point& other) const { return !((*this) == other); }

  // getter.
  T get_x() const { return _x; }
  T get_y() const { return _y; }

  // setter.
  void set_x(const T& x) { _x = x; }
  void set_y(const T& y) { _y = y; }

  bool isUnLegal() const { return _x == -1 && _y == -1; }

 private:
  T _x;
  T _y;
};
}  // namespace ipl

#endif