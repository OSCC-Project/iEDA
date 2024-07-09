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

#include <iostream>
#include <map>
#include <vector>

#include "Point.h"
#include "Rects.h"
#include "define.h"
#include "flute3/flute.h"
#include "timing_engine.h"
#include "ToConfig.h"

namespace ito {

using XCord = int;  // x-axis coordinates
using YCord = int;  // y-axis coordinates

class TreeBuild;
class Node;


class Node {
 public:
  Node() = default;
  Node(Point pt) : _pt(pt) {}
  ~Node() = default;

  void set_father_node(Node *f_n);
  void set_first_child(Node *f_c) { _first_child = f_c; }
  void set_next_sibling(Node *n_s) { _next_sibling = n_s; }
  void set_id(int id) { _id = id; }

  Node *get_father_node() const { return _father_node; }
  Node *get_first_child() const { return _first_child; }
  Node *get_next_sibling() const { return _next_sibling; }
  int   get_id() const { return _id; }
  Point get_location() { return _pt; }

  bool operator==(const Node n2) { return this->_pt == n2._pt; }
  int  get_child_num() const { return get_children().size(); }

  std::vector<Node *> get_children() const;

  void print_recursive(std::ostream &os) const;
 private:
  int   _id = -1;
  Point _pt;
  Node *_father_node = nullptr;
  Node *_first_child = nullptr;
  Node *_next_sibling = nullptr;

  // void print_single(std::ostream &os) const;
  void print_single(std::ostream &os, std::vector<bool> &prefix) const;
};

class TreeBuild
{
 public:
  TreeBuild() {}
  ~TreeBuild() = default;

  // set
  void set_root(Node* root) { this->_root = root; }
  void set_points(std::vector<Point> pts) { this->_points = pts; }

  // get
  TODesignObjSeq& get_pins() { return _pins; }
  Node* get_root() const { return _root; }
  std::vector<Point> get_points() const { return _points; }
  Point get_location(int idx) const { return _points[idx]; }
  unsigned int get_pins_count() { return _pins.size(); }

  DesignObject* get_pin(int idx) const;

  bool makeRoutingTree(ista::Net* net, RoutingType rout_type);

  void build(RoutingType rout_type, int x[], int y[], int pin_num, int driver_id);

  void segmentIndexAndLength(Node* root, std::vector<std::pair<int, int>>& wire_segment_idx, std::vector<int>& length);
  void driverToLoadLength(Node* root, std::vector<int>& load_index, std::vector<int>& length, int curr_dist);
  void updateBranch();
  // require updateBranch
  int left(int idx) const { return _left_branch[idx]; }
  int middle(int idx) const { return _middle_branch[idx]; }
  int right(int idx) const { return _right_branch[idx]; }

  std::vector<std::vector<int>> findPathOverMaxLength(Node* root, int sum, int max_length, std::vector<std::vector<int>>& res);

  void set_pin_visit(int idx) { _is_visit_pin[idx] = 1; }
  int get_pin_visit(int idx) { return _is_visit_pin[idx]; }

  void plotConectionInfo(const std::string file_name, const std::string net_name, TreeBuild* tree, Rectangle core);

  static int _null_pt;
  void Print(std::ostream& os = std::cout) const;
  friend std::ostream& operator<<(std::ostream& os, const TreeBuild* tree)
  {
    tree->Print(os);
    return os;
  }

  // void convertToBinaryTree() { convertToBinaryTree(_root); }

 private:
  
  void makeHVTree(int x[], int y[], int pin_count, int driver_id);
  void makeSteinerTree(int x[], int y[], int pin_count, int driver_id);
  void makeShallowLightTree(int x[], int y[], int pin_count, int driver_id);

  void preOrderTraversal(Node* root);

  void deepFirstSearch(Node* root, int sum, std::vector<std::vector<int>>& res, std::vector<int> vec, int max_length);

  void plotPinName(std::stringstream& feed, const std::string net_name, TreeBuild* tree, int id);

  void convertToBinaryTree(Node* root);

  // _root always cooresponds to the driver pin.
  Node* _root = nullptr;
  TODesignObjSeq _pins;

  // pin pts + steiner pts, the index of steiner pts is behind of the pin pts.
  std::vector<Point> _points;

  // Since each node has at most three branches, due to the way of TreeBuild is
  // constructed. _left_branch[i] = j means: node j is the left child of node i.
  std::vector<int> _left_branch;
  std::vector<int> _middle_branch;
  std::vector<int> _right_branch;

  std::vector<int> _is_visit_pin;
};
}  // namespace ito