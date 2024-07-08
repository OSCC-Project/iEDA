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
#include <boost/functional/hash.hpp>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "../TreeBuild.h"
#include "salt/salt.h"
namespace ito {

class ShallowLightTree
{
 public:
  ShallowLightTree() = default;
  ~ShallowLightTree() = default;

  Node* makeShallowLightTree(int x[], int y[], int pin_count, int driver_id);

  std::vector<Point> getPoints() { return _points; }

 private:
  void findLeftRight(Node *father, const std::shared_ptr<salt::TreeNode>& salt_node);

  std::vector<Point> _points;
};

}  // namespace ito