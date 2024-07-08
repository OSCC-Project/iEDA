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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "../TreeBuild.h"

namespace ito {

class HVTree
{
 public:
  HVTree() = default;
  ~HVTree() = default;

  Node* makeHVTree(int x[], int y[], int pin_count, int driver_id);

  std::vector<Point> getPoints() { return _points; }

 private:
  void insertTree(Node* father, Node* child);
  int findIndexInPoints(Node* n);
  void updateTree(int index, XCord x_coordinate, Node*& parent, std::vector<int> cordinate_vec,
                  std::multimap<YCord, XCord> cordinate_to_x_map);

  std::vector<Point> _points;
};

}  // namespace ito
