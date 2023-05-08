#pragma once

#include "Traits.h"
#include "pgl.h"

namespace icts {

template <typename T>
class DmeNode {
 public:
  DmeNode() : _data() {}
  DmeNode(const T &data) : _data(data) {}
  DmeNode(const DmeNode &) = default;

  T &get_data() { return _data; }
  void set_data(const T &data) { _data = data; }

  Point get_loc() const {
    auto x = DataTraits<T>::getX(_data);
    auto y = DataTraits<T>::getY(_data);
    return Point(x, y);
  }
  void set_loc(const Point &loc) {
    DataTraits<T>::setX(_data, loc.x());
    DataTraits<T>::setY(_data, loc.y());
  }

 private:
  T _data;
};

}  // namespace icts