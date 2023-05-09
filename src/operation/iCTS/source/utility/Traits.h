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

namespace icts {

template <typename T>
struct DataTraits {
  typedef typename T::coordinate_type coordinate_type;
  typedef typename T::point_type point_type;
  typedef typename T::id_type id_type;

  static inline id_type getId(const T &t) { return t.getId(); }
  static inline coordinate_type getX(const T &t) { return t.getX(); }
  static inline coordinate_type getY(const T &t) { return t.getY(); }
  static inline point_type getPoint(const T &t) { return t.getPoint(); }
  static inline id_type getSubWirelength(const T &t) {
    return t.getSubWirelength();
  }
  static inline void setId(T &t, id_type id) { t.setId(id); }
  static inline void setX(T &t, coordinate_type x) { t.setX(x); }
  static inline void setY(T &t, coordinate_type y) { t.setY(y); }
  static inline void setPoint(T &t, const point_type &p) { t.setPoint(p); }
};

template <typename T>
struct TimeTraits {
  static inline double getTime(const T &t) { return t.getTime(); }
  static inline void setTime(T &t, double time) { t.setTime(time); }
};

}  // namespace icts