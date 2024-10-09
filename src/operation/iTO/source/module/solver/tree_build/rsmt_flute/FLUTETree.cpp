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

#include "FLUTETree.h"

#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace ito {
Node* FLUTETree::makeFLUTETree(int x[], int y[], int pin_count, int driver_id)
{
  for (int i = 0; i < pin_count; i++) {
    Point pin_loc = Point(x[i], y[i]);
    _points.push_back(pin_loc);
    _pin_location_2_id[std::make_pair(x[i], y[i])] = i;
  }
  // steiner tree
  Flute::Tree stTree;

  stTree = Flute::flute(pin_count, x, y, FLUTE_ACCURACY);
  // Flute::printtree(stTree);
  // Find driver steiner point.
  XCord driver_x_coordinate = x[driver_id];
  YCord driver_y_coordinate = y[driver_id];
  Node* _root = new Node(Point(driver_x_coordinate, driver_y_coordinate));

  int driver_id_in_flute = 0;  // flute生成的topology会进行重新排序，所以需要找到driver在flute中的id
  int branch_count = stTree.deg * 2 - 2;
  _points.resize(branch_count);
  std::vector<std::vector<int>> adj(branch_count);
  bool have_find_driver_id_in_flute = false;
  for (int i = 0; i < branch_count; i++) {
    Flute::Branch& branch_pt = stTree.branch[i];
    int j = branch_pt.n;
    // printf("(%d, %d)\n", i, j);
    if (i != j) {
      adj[i].push_back(j);
      adj[j].push_back(i);
    }

    if (!have_find_driver_id_in_flute) {
      if (branch_pt.x == driver_x_coordinate && branch_pt.y == driver_y_coordinate) {
        driver_id_in_flute = i;
        have_find_driver_id_in_flute = true;
      }
    }

    if (i >= stTree.deg) {
      Point pt = Point(stTree.branch[i].x, stTree.branch[i].y);
      _points[i] = pt;
    }
  }

  _root->set_id(driver_id);

  findLeftRight(_root, -1, driver_id_in_flute, adj, stTree);
  return _root;
}

void FLUTETree::findLeftRight(Node* father, int father_id, int child_id, std::vector<std::vector<int>>& adj, Flute::Tree stTree)
{
  auto insertTree = [&](Node* father, Node* child) {
    Node* first_child = father->get_first_child();
    if (first_child == nullptr) {
      // father->set_first_child(child);
      child->set_father_node(father);
    } else {
      while (first_child->get_next_sibling()) {
        first_child = first_child->get_next_sibling();
      }
      // first_child->set_next_sibling(child);
      child->set_father_node(father);
    }
  };

  Node* child = nullptr;
  for (int j : adj[child_id]) {
    Point pt = Point(stTree.branch[j].x, stTree.branch[j].y);
    if (j != father_id) {
      child = new Node(pt);
      insertTree(father, child);
      if (j < stTree.deg) {
        auto it = _pin_location_2_id.find({pt.get_x(), pt.get_y()});
        if (it == _pin_location_2_id.end()) {
          std::cout << "Can't find target pt in Points" << std::endl;
          exit(1);
        } else {
          child->set_id(it->second);
        }
      } else {
        child->set_id(j);
      }
      // cout << child_id << " , " << j << endl;
      findLeftRight(child, child_id, j, adj, stTree);
    }
  }
}
}  // namespace ito