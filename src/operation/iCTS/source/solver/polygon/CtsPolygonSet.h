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