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

#include "ShallowLightTree.h"

#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace ito {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

std::vector<std::pair<int, int>> findDuplicateCoordinates(const int real_x[], const int real_y[], int size) {
    std::vector<std::pair<int, int>> result;
    std::unordered_map<std::pair<int, int>, int, pair_hash> coord_map;

    for (size_t i = 0; i < size; ++i) {
        std::pair<int, int> coord = {real_x[i], real_y[i]};
        if (coord_map.find(coord) != coord_map.end()) {
            result.push_back({coord_map[coord], static_cast<int>(i)});
        } else {
            coord_map[coord] = static_cast<int>(i);
        }
    }

    return result;
}

Node* ShallowLightTree::makeShallowLightTree(int x[], int y[], int pin_count, int driver_id)
{
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;

  XCord driver_x_coordinate = x[driver_id];
  YCord driver_y_coordinate = y[driver_id];
  Node* _root = new Node(Point(driver_x_coordinate, driver_y_coordinate));

  auto check = findDuplicateCoordinates(x, y, pin_count);
    for (auto c : check) {
      x[c.first] += 1;
    }

  for (int i = 0; i < pin_count; ++i) {
    auto salt_pin = std::make_shared<salt::Pin>(x[i], y[i], i, 0);
    salt_pins.push_back(salt_pin);
  }

  // run SALT
  salt::Net net;
  net.init(0, "", salt_pins);
  salt::Tree tree;
  salt::SaltBuilder salt_builder;
  salt_builder.Run(net, tree, 0);

  auto source = tree.source;
  int source_id = source->id;
  _root->set_id(source_id);

  _points.resize(2 * pin_count);

  findLeftRight(_root, source);

  return _root;
}

void ShallowLightTree::findLeftRight(Node* father, const std::shared_ptr<salt::TreeNode>& salt_node)
{
  auto insert_tree = [&](Node* father, Node* child, int id_child) {
    Node* first_child = father->get_first_child();
    if (first_child == nullptr) {
      child->set_father_node(father);

      child->set_id(id_child);

    } else {
      while (first_child->get_next_sibling()) {
        first_child = first_child->get_next_sibling();
      }
      child->set_father_node(father);

      child->set_id(id_child);
    }
  };

  _points[salt_node->id] = (Point(salt_node->loc.x, salt_node->loc.y));

  Node* child = nullptr;
  for (auto child_node : salt_node->children) {
    Point pt = Point(child_node->loc.x, child_node->loc.y);

    child = new Node(pt);

    insert_tree(father, child, child_node->id);
    findLeftRight(child, child_node);
  }
}
}  // namespace ito