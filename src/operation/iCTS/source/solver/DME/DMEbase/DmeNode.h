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