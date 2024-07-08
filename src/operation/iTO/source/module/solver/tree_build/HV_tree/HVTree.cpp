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

#include "HVTree.h"

namespace ito {

Node* HVTree::makeHVTree(int x[], int y[], int pin_count, int driver_id)
{
  for (int i = 0; i < pin_count; i++) {
    Point pin_loc = Point(x[i], y[i]);
    _points.push_back(pin_loc);
  }

  std::multimap<YCord, XCord> y_to_x_map;
  // Determine how many intermediate nodes should be inserted.
  std::set<YCord> coordinate_set;
  for (int i = 0; i != pin_count; ++i) {
    coordinate_set.insert(y[i]);

    std::pair<YCord, XCord> tmp(y[i], x[i]);
    y_to_x_map.insert(tmp);
  }
  // Calc the x-axis center of gravity of all pins
  long long int sum = 0;
  for (int i = 0; i != pin_count; ++i) {
    sum += x[i];
  }
  XCord x_gravity = sum / pin_count;

  std::vector<YCord> y_cordinate_vec;
  for (YCord coordinate : coordinate_set) {
    _points.push_back(Point(x_gravity, coordinate));  // add steiner point behind "_points"
    y_cordinate_vec.push_back(coordinate);
  }

  XCord driver_x_coordinate = x[driver_id];  // _root->get_location().get_x();
  YCord driver_y_coordinate = y[driver_id];  // _root->get_location().get_y();
  Node* _root = new Node(Point(driver_x_coordinate, driver_y_coordinate));

  int pin_count_curr_y = y_to_x_map.count(driver_y_coordinate);
  std::multimap<int, int>::iterator iter_curr;
  iter_curr = y_to_x_map.find(driver_y_coordinate);

  // "x_gravity" is exactly the x-coordinate of a pin
  bool x_gravity_at_pin = false;
  std::vector<XCord> pin_x_cord;
  for (int i = 0; i < pin_count_curr_y; i++, ++iter_curr) {
    pin_x_cord.push_back(iter_curr->second);
    if (x_gravity == iter_curr->second) {
      x_gravity_at_pin = true;
    }
  }
  if (!x_gravity_at_pin) {
    pin_x_cord.push_back(x_gravity);
    std::sort(pin_x_cord.begin(), pin_x_cord.end());
    pin_count_curr_y += 1;
  }

  // find the index of driver pin
  std::vector<XCord>::iterator itr = find(pin_x_cord.begin(), pin_x_cord.end(), driver_x_coordinate);
  int driver_idx = distance(pin_x_cord.begin(), itr);
  int idx_begin_right = driver_idx + 1;
  int idx_begin_left = driver_idx - 1;
  Node* first_steiner_pt = nullptr;

  // go right
  Node* parent1 = _root;
  while (idx_begin_right < pin_count_curr_y) {
    Point p_child = Point(pin_x_cord[idx_begin_right], driver_y_coordinate);
    Node* child = new Node(p_child);
    insertTree(parent1, child);
    if (pin_x_cord[idx_begin_right] == x_gravity
        // steiner pt not at driver pin location
        && x_gravity != driver_x_coordinate) {
      first_steiner_pt = child;
    }
    parent1 = child;
    idx_begin_right++;
  }
  // go left
  Node* parent2 = _root;
  while (idx_begin_left >= 0) {
    Point p_child2 = Point(pin_x_cord[idx_begin_left], driver_y_coordinate);
    Node* child2 = new Node(p_child2);
    insertTree(parent2, child2);
    if (pin_x_cord[idx_begin_left] == x_gravity && x_gravity != driver_x_coordinate && first_steiner_pt == nullptr) {
      first_steiner_pt = child2;
    }
    parent2 = child2;
    idx_begin_left--;
  }

  if (first_steiner_pt != nullptr) {
    int id = findIndexInPoints(_root);
    _root->set_id(id);
  } else {
    first_steiner_pt = _root;
    int id = findIndexInPoints(_root);
    _root->set_id(id);
  }

  std::vector<int>::iterator it = find(y_cordinate_vec.begin(), y_cordinate_vec.end(), driver_y_coordinate);
  auto index = std::distance(std::begin(y_cordinate_vec), it);
  int index2 = index;

  ///////////////////////////////////////////////////
  Node* temp_node = first_steiner_pt;
  while (index > 0) {
    index--;
    updateTree(index, x_gravity, temp_node, y_cordinate_vec, y_to_x_map);
  }
  Node* temp_node2 = first_steiner_pt;
  int number_of_intermediate_points = y_cordinate_vec.size() - 1;
  while (index2 < number_of_intermediate_points) {
    index2++;
    updateTree(index2, x_gravity, temp_node2, y_cordinate_vec, y_to_x_map);
  }
  return _root;
}

/**
 * @brief Insert a child node of the parent node, and update node's info
 *
 * @param father
 * @param child
 */
void HVTree::insertTree(Node* father, Node* child)
{
  Node* first_child = father->get_first_child();
  if (first_child == nullptr) {
    // father->set_first_child(child);
    child->set_father_node(father);

    int id = findIndexInPoints(child);
    child->set_id(id);

  } else {
    while (first_child->get_next_sibling()) {
      first_child = first_child->get_next_sibling();
    }
    // first_child->set_next_sibling(child);

    child->set_father_node(father);

    int id = findIndexInPoints(child);
    child->set_id(id);
  }
}

/**
 * @brief Return the index of node in the member "_points"
 */
int HVTree::findIndexInPoints(Node* n)
{
  if (n->get_id() != -1) {
    return n->get_id();
  }
  auto target_pt = n->get_location();
  int num_of_pt = _points.size();
  for (int i = 0; i < num_of_pt; ++i) {
    if (target_pt == _points[i] && !_points[i].isVisit()) {
      n->set_id(i);
      _points[i].set_is_visit();
      return i;
    }
  }
  std::cout << "Can't find target pt in Points" << std::endl;
  exit(1);
  return 0;
}

/**
 * @brief
 *
 * @param index Index of the y-coordinate of the Steiner point in param "x_coordinate"
 * @param x_coordinate The x-coordinates of all Steiner point
 * @param parent parent node
 * @param cordinate_vec y-coordinate of all Steiner point
 * @param cordinate_to_x_map x-coordinate of nodes with the same y-coordinate
 */
void HVTree::updateTree(int index, XCord x_coordinate, Node*& parent, std::vector<int> cordinate_vec,
                        std::multimap<YCord, XCord> cordinate_to_x_map)
{
  YCord cur_y = cordinate_vec[index];
  Node* father = new Node(Point(x_coordinate, cur_y));
  insertTree(parent, father);
  parent = father;

  int pin_count_curr_y = cordinate_to_x_map.count(cur_y);
  std::multimap<int, int>::iterator iter_curr;
  iter_curr = cordinate_to_x_map.find(cur_y);

  bool excute_once = true;
  // "x_gravity" is exactly the x-coordinate of a pin
  bool x_gravity_at_pin = false;
  int idx_begin_right = pin_count_curr_y - 1;
  std::vector<XCord> pin_x_cord;
  for (int i = 0; i < pin_count_curr_y; i++, ++iter_curr) {
    pin_x_cord.push_back(iter_curr->second);
    if (excute_once) {
      if (x_coordinate <= iter_curr->second) {
        excute_once = false;
        idx_begin_right = i;
        if (x_coordinate == iter_curr->second) {
          x_gravity_at_pin = true;
        }
      }
    }
  }
  int idx_begin_left = idx_begin_right - 1;
  // x_coordinate happens to be the horizontal coordinate of a pin
  // and needs to be changed from the starting point of the index
  if (x_gravity_at_pin) {
    idx_begin_right = idx_begin_right + 1;
  }
  Node* parent2 = father;
  // go right
  while (idx_begin_right < pin_count_curr_y) {
    Point p_child = Point(pin_x_cord[idx_begin_right], cur_y);
    Node* child = new Node(p_child);
    insertTree(father, child);

    father = child;
    idx_begin_right++;
  }
  // go left
  while (idx_begin_left >= 0) {
    Point p_child = Point(pin_x_cord[idx_begin_left], cur_y);
    Node* child = new Node(p_child);
    insertTree(parent2, child);

    parent2 = child;
    idx_begin_left--;
  }
}
}  // namespace ito
