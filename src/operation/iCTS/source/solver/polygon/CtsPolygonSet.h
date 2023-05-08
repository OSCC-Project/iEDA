#pragma once
#include <vector>

#include "CtsPolygon.h"

namespace icts {

template <typename T>
class CtsPolygonSet : public std::vector<CtsPolygon<T>> {
 public:
  typedef CtsPolygon<T> value_type;
  typedef typename std::vector<CtsPolygon<T>>::const_iterator iterator_type;

  template <typename PolySet>
  CtsPolygonSet &operator=(const PolySet &polyset) {
    for (auto itr = polyset.begin(); itr != polyset.end(); ++itr) {
      push_back(*itr);
    }
    return *this;
  }
};

}  // namespace icts