#ifndef IDRC_SRC_DB_COORDINATE_H_
#define IDRC_SRC_DB_COORDINATE_H_

#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <vector>

namespace idrc {

template <typename T>
class DrcCoordinate
{
 public:
  explicit DrcCoordinate() {}
  explicit DrcCoordinate(const T x, const T y)
  {
    static_assert(std::is_same<T, int>::value || std::is_same<T, double>::value, "Must use int or double here!");
    _x = x;
    _y = y;
  }
  DrcCoordinate(const DrcCoordinate& other)
  {
    _x = other._x;
    _y = other._y;
  }
  DrcCoordinate(DrcCoordinate&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
  }
  ~DrcCoordinate() {}
  DrcCoordinate& operator=(const DrcCoordinate& other)
  {
    _x = other._x;
    _y = other._y;
    return (*this);
  }
  DrcCoordinate& operator=(DrcCoordinate&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
    return (*this);
  }
  bool operator==(const DrcCoordinate& other) { return (_x == other._x && _y == other._y); }
  bool operator!=(const DrcCoordinate& other) { return !((*this) == other); }
  // getter
  T get_x() const { return _x; }
  T get_y() const { return _y; }
  // setters
  void set(const T& x, const T& y)
  {
    _x = x;
    _y = y;
  }
  void set(DrcCoordinate& in)
  {
    _x = in.get_x();
    _y = in.get_y();
  }
  void set_x(const T& x) { _x = x; }
  void set_y(const T& y) { _y = y; }
  // function

 private:
  T _x = -1;
  T _y = -1;
};

template <typename T>
struct cmpCoordinateXASC
{
  bool operator()(const DrcCoordinate<T>& a, const DrcCoordinate<T>& b) const
  {
    return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_y() < b.get_y();
  }
};

template <typename T>
struct cmpCoordinateYASC
{
  bool operator()(const DrcCoordinate<T>& a, const DrcCoordinate<T>& b) const
  {
    return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_x() < b.get_x();
  }
};

}  // namespace idrc
#endif  // IDRC_SRC_DB_COORDINATE_H_
