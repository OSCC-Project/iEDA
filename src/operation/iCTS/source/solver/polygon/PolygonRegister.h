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
#include <boost/polygon/gtl.hpp>

#include "pgl.h"

namespace boost {
namespace polygon {

template <typename T>
struct geometry_concept<icts::CtsInterval<T>> {
  typedef interval_concept type;
};

// point type
template <typename T>
struct geometry_concept<icts::CtsPoint<T>> {
  typedef point_concept type;
};

// segment type
template <typename T>
struct geometry_concept<icts::CtsSegment<T>> {
  typedef segment_concept type;
};

// rectangle type
template <typename T>
struct geometry_concept<icts::CtsRectangle<T>> {
  typedef rectangle_concept type;
};

// polygon type
template <typename T>
struct geometry_concept<icts::CtsPolygon<T>> {
  typedef polygon_concept type;
};

template <typename T>
struct geometry_concept<icts::CtsPolygonSet<T>> {
  typedef polygon_set_concept type;
};

template <typename T>
struct polygon_set_traits<icts::CtsPolygonSet<T>> {
  typedef typename icts::CtsPolygon<T>::coord_t coordinate_type;
  typedef typename icts::CtsPolygon<T>::point_t point_type;
  typedef typename icts::CtsPolygonSet<T>::iterator_type iterator_type;

  static inline iterator_type begin(const icts::CtsPolygonSet<T> &poly_set) {
    return poly_set.begin();
  }
  static inline iterator_type end(const icts::CtsPolygonSet<T> &poly_set) {
    return poly_set.end();
  }
  static inline bool clean(const icts::CtsPolygonSet<T> &poly_set) {
    return false;
  }
  static inline bool sorted(const icts::CtsPolygonSet<T> &poly_set) {
    return false;
  }
};

template <typename T>
struct polygon_set_mutable_traits<icts::CtsPolygonSet<T>> {
  typedef T coordinate_type;

  template <typename input_iterator_type>
  static inline void set(icts::CtsPolygonSet<T> &poly_set,
                         input_iterator_type input_begin,
                         input_iterator_type input_end) {
    poly_set.clear();

    polygon_set_data<coordinate_type> ps;
    ps.insert(input_begin, input_end);
    ps.get(poly_set);
  }
};

}  // namespace polygon
}  // namespace boost
