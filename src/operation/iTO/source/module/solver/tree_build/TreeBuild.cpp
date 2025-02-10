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

#include "TreeBuild.h"

#include "FLUTETree.h"
#include "HVTree.h"
#include "ShallowLightTree.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ito {
int TreeBuild::_null_pt = -1;

std::vector<Node*> Node::get_children() const
{
  std::vector<Node*> children;
  if (!_first_child) {
    return children;
  }
  children.push_back(_first_child);
  auto first_child = get_first_child();
  while (first_child->get_next_sibling()) {
    first_child = first_child->get_next_sibling();
    children.push_back(first_child);
  }
  return children;
}

void Node::set_father_node(Node* f_n)
{
  _father_node = f_n;
  if (_father_node->get_first_child() != nullptr) {
    Node* tmp = _father_node->get_first_child();
    if (tmp->get_id() == _id) return;
    while (tmp->get_next_sibling() != nullptr) {
      tmp = tmp->get_next_sibling();
      if (tmp->get_id() == _id) return;
    }
    tmp->set_next_sibling(this);
  } else {
    _father_node->set_first_child(this);
  }
}

void Node::print_recursive(std::ostream& os) const
{
  std::vector<bool> prefix;
  print_single(os, prefix);
}

void Node::print_single(std::ostream& os, std::vector<bool>& prefix) const
{
  for (auto pre : prefix)
    os << (pre ? "  |" : "   ");
  if (!prefix.empty())
    os << "-> ";
  int child_num = get_child_num();
  os << "Node " << _id << ": " << _pt << ", " << child_num << " children";
  os << std::endl;
  auto children = get_children();
  if (children.size() > 0) {
    prefix.push_back(true);
    for (size_t i = 0; i < children.size() - 1; ++i) {
      if (children[i])
        children[i]->print_single(os, prefix);
      else
        os << "<null>" << std::endl;
    }
    prefix.back() = false;
    children.back()->print_single(os, prefix);
    prefix.pop_back();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
bool TreeBuild::makeRoutingTree(ista::Net* net, RoutingType rout_type)
{
  auto getConnectedPins = [](ista::Net* net, TODesignObjSeq& pins) {
    pins.clear();
    DesignObject* obj;
    FOREACH_NET_PIN(net, obj)
    {
      pins.push_back(obj);
    }
  };

  struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& pair) const
    {
      return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second);
    }
  };

  auto findDuplicateCoordinates = [](const int* real_x, const int* real_y, int size) {
    std::unordered_map<int, std::vector<int>> result_map;
    std::unordered_map<std::pair<int, int>, int, pair_hash> coord_map;

    for (int i = 0; i < size; ++i) {
      std::pair<int, int> coord = {real_x[i], real_y[i]};
      if (coord_map.find(coord) != coord_map.end()) {
        if (result_map.find(coord_map[coord]) != result_map.end()) {
          result_map[coord_map[coord]].push_back(i);
        } else {
          result_map[coord_map[coord]] = {i};
        }
      } else {
        coord_map[coord] = static_cast<int>(i);
      }
    }
    return result_map;
  };

  getConnectedPins(net, _pins);
  int pin_num = _pins.size();
  if (pin_num < 2) {
    return false;  // nullptr;
  }

  int* x = new int[pin_num];
  int* y = new int[pin_num];
  int driver_id = 0;
  for (int i = 0; i < pin_num; i++) {
    DesignObject* pin = _pins[i];

    auto idb_loc = timingEngine->get_sta_adapter()->idbLocation(pin);
    Point pin_loc = Point(idb_loc->get_x(), idb_loc->get_y());
    x[i] = pin_loc.get_x();
    y[i] = pin_loc.get_y();

    if (pin_loc.get_x() == -1 && pin_loc.get_y() == -1) {
      return false;
    }

    if (pin->isOutput()) {
      driver_id = i;
    }
  }
  std::swap(_pins[0], _pins[driver_id]);
  std::swap(x[0], x[driver_id]);
  std::swap(y[0], y[driver_id]);

  // 对重合的节点坐标进行微调
  auto check = findDuplicateCoordinates(x, y, pin_num);
  for (auto c : check) {
    for (int i = 0; i < c.second.size(); i++) {
      x[c.second[i]] += (i+1);
    }
  }

  driver_id = 0;
  build(rout_type, x, y, pin_num, driver_id);
  return true;
}

void TreeBuild::build(RoutingType rout_type, int x[], int y[], int pin_num, int driver_id)
{
  switch (rout_type) {
    case RoutingType::kHVTree:
      makeHVTree(x, y, pin_num, driver_id);
      break;
    case RoutingType::kSteiner:
      makeSteinerTree(x, y, pin_num, driver_id);
      break;
    case RoutingType::kShallowLight:
      makeShallowLightTree(x, y, pin_num, driver_id);
      break;
    default:
      break;
  }
}

void TreeBuild::plotConectionInfo(const std::string file_name, const std::string net_name, TreeBuild* tree, Rectangle core)
{
  std::string path = "/output/";

  std::ofstream dot_connection(path + file_name + ".gds");
  if (!dot_connection.good()) {
    std::cerr << "write_connection:: cannot open `" << file_name << "' for writing." << std::endl;
    exit(1);
  }

  std::stringstream feed;

  feed << "HEADER 600" << std::endl;
  feed << "BGNLIB" << std::endl;
  feed << "LIBNAME ITDP_LIB" << std::endl;
  feed << "UNITS 0.001 1e-9" << std::endl;
  //
  feed << "BGNSTR" << std::endl;
  feed << "STRNAME core" << std::endl;
  feed << "BOUNDARY" << std::endl;
  feed << "LAYER 0" << std::endl;
  feed << "DATATYPE 0" << std::endl;
  feed << "XY" << std::endl;

  feed << core.get_x_min() << " : " << core.get_y_min() << std::endl;
  feed << core.get_x_max() << " : " << core.get_y_min() << std::endl;
  feed << core.get_x_max() << " : " << core.get_y_max() << std::endl;
  feed << core.get_x_min() << " : " << core.get_y_max() << std::endl;
  feed << core.get_x_min() << " : " << core.get_y_min() << std::endl;
  feed << "ENDEL" << std::endl;

  // tmp set the wire width.
  int64_t wire_width = 160;

  std::vector<std::pair<int, int>> wire_segment_idx;
  std::vector<int> length_per_wire;

  tree->segmentIndexAndLength(tree->get_root(), wire_segment_idx, length_per_wire);
  int numb = wire_segment_idx.size();
  for (int i = 0; i != numb; ++i) {
    int index1 = wire_segment_idx[i].first;
    plotPinName(feed, net_name, tree, index1);
    int index2 = wire_segment_idx[i].second;
    plotPinName(feed, net_name, tree, index2);

    // plot wire.
    feed << "PATH" << std::endl;
    feed << "LAYER 2" << std::endl;
    feed << "DATATYPE 0" << std::endl;
    feed << "WIDTH " + std::to_string(wire_width) << std::endl;
    feed << "XY" << std::endl;
    feed << tree->get_location(index1).get_x() << " : " << tree->get_location(index1).get_y() << std::endl;
    feed << tree->get_location(index2).get_x() << " : " << tree->get_location(index2).get_y() << std::endl;
    feed << "ENDEL" << std::endl;
  }
  feed << "ENDSTR" << std::endl;
  feed << "ENDLIB" << std::endl;
  dot_connection << feed.str();
  feed.clear();
  dot_connection.close();
}

void TreeBuild::plotPinName(std::stringstream& feed, const std::string net_name, TreeBuild* tree, int id)
{
  feed << "TEXT" << std::endl;
  feed << "LAYER 1" << std::endl;
  feed << "TEXTTYPE 0" << std::endl;
  feed << "PRESENTATION 0,2,0" << std::endl;
  feed << "PATHTYPE 1" << std::endl;
  feed << "STRANS 0,0,0" << std::endl;
  feed << "MAG 1875" << std::endl;
  feed << "XY" << std::endl;
  auto loc = tree->get_location(id);
  int pin_x = loc.get_x();
  int pin_y = loc.get_y();
  feed << pin_x << " : " << pin_y << std::endl;

  if (tree->get_pin(id)) {  // the real pin.
    const char* name = tree->get_pin(id)->get_name();
    string str(name);
    feed << "STRING " + str << std::endl;
  } else {  // the stainer pin.
    feed << "STRING " + net_name + " : " + std::to_string(id) << std::endl;
  }
  feed << "ENDEL" << std::endl;
}

static int manhattanDistance(Node* n0, Node* n1)
{
  Point p0 = n0->get_location();
  Point p1 = n1->get_location();
  int x0 = p0.get_x();
  int x1 = p1.get_x();

  int y0 = p0.get_y();
  int y1 = p1.get_y();

  return abs(x0 - x1) + abs(y0 - y1);
}

/**
 * @brief return max distance from driver pin to load pins.
 */
void findMaxDistfromDrvr(Node* root, int curr_dist, int& max_dist)
{
  Node* first_child = root->get_first_child();

  if (first_child) {
    int dis = manhattanDistance(first_child, first_child->get_father_node());
    int sum = curr_dist + dis;
    findMaxDistfromDrvr(first_child, sum, max_dist);
  } else {
    max_dist = curr_dist > max_dist ? curr_dist : max_dist;
  }

  Node* child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
    int dis1 = manhattanDistance(child, child->get_father_node());  /////
    int dis = manhattanDistance(root, root->get_father_node());
    int sum2 = curr_dist - dis + dis1;
    findMaxDistfromDrvr(child, sum2, max_dist);
  }
}

/**
 * @brief length from driver pin to all load pins
 */
void TreeBuild::driverToLoadLength(Node* root, std::vector<int>& load_index, std::vector<int>& length, int curr_dist)
{
  Node* first_child = root->get_first_child();
  if (first_child) {
    int dis = manhattanDistance(first_child, first_child->get_father_node());
    int sum = curr_dist + dis;
    if (first_child->get_id() < (int) _pins.size()) {
      load_index.push_back(first_child->get_id());
      length.push_back(sum);
    }
    driverToLoadLength(first_child, load_index, length, sum);
  }

  Node* child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
    int dis1 = manhattanDistance(child, child->get_father_node());  /////
    int dis = manhattanDistance(root, root->get_father_node());
    int sum = curr_dist - dis + dis1;
    if (child->get_id() < (int) _pins.size()) {
      load_index.push_back(child->get_id());
      length.push_back(sum);
    }
    driverToLoadLength(child, load_index, length, sum);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void TreeBuild::Print(std::ostream& os) const
{
  os << "Tree ";
  os << "#pins=" << _pins.size() << std::endl;

  _root->print_recursive(os);
}

void TreeBuild::makeHVTree(int x[], int y[], int pin_count, int driver_id)
{
  /*
       (x1,y3)    (x_g,y3)          (x2,y3)                                           drv
          o-----------o-pt3------------o                                               |
        load4         |              load3                                            pt1
                      |                                                                |
       (x1,y2)    (x_g,y2)               (x3,y2)    (x4,y2)                           pt2
          o-----------o-pt2-----------------o---------o                             /  | \
        load2         |                   load1      load5                         /   | \
                      |                                                           /    | \
   (x0,y1)        (x_g,y1)                                                    load1  load2
   pt3
      o---------------o pt1                                                     / /   \
      drv                                                                    load5 load3
   load4
  */

  HVTree hv_tree;
  _root = hv_tree.makeHVTree(x, y, pin_count, driver_id);
  _points = hv_tree.getPoints();
  _is_visit_pin.resize(_points.size(), 0);
  convertToBinaryTree(_root);
}

void TreeBuild::makeSteinerTree(int x[], int y[], int pin_count, int driver_id)
{
  FLUTETree flute_tree;
  _root = flute_tree.makeFLUTETree(x, y, pin_count, driver_id);
  _points = flute_tree.getPoints();
  _is_visit_pin.resize(_points.size(), 0);
  convertToBinaryTree(_root);
}

void TreeBuild::makeShallowLightTree(int x[], int y[], int pin_count, int driver_id)
{
  ShallowLightTree sl_tree;
  _root = sl_tree.makeShallowLightTree(x, y, pin_count, driver_id);
  _points = sl_tree.getPoints();
  _is_visit_pin.resize(_points.size(), 0);
  convertToBinaryTree(_root);
}

DesignObject* TreeBuild::get_pin(int idx) const
{
  int pin_count = _pins.size();
  if (idx < pin_count) {
    return _pins[idx];
  } else {
    return nullptr;
  }
}

/**
 * @brief Update the branches(left/middle/right) of each node.
 *
 * This function needs to be executed before using each branch of the node
 */
void TreeBuild::updateBranch()
{
  int numb_pts = _points.size();
  _left_branch.resize(numb_pts, _null_pt);
  _middle_branch.resize(numb_pts, _null_pt);
  _right_branch.resize(numb_pts, _null_pt);
  preOrderTraversal(_root);
  // printf("      ");
  // for (int i = 0; i < numb_pts; i++) {
  //   printf("  %-2d", i);
  // }
  // printf("\nleft  ");
  // for (int i = 0; i < numb_pts; i++) {
  //   printf("  %-2d", _left_branch[i]);
  // }
  // printf("\nmiddle");
  // for (int i = 0; i < numb_pts; i++) {
  //   printf("  %-2d", _middle_branch[i]);
  // }
  // printf("\nright ");
  // for (int i = 0; i < numb_pts; i++) {
  //   printf("  %-2d", _right_branch[i]);
  // }
  // printf("\n");
}

/**
 * @brief Pre-order traversal updates the branches of each node
 *
 * @param root
 */
void TreeBuild::preOrderTraversal(Node* root)
{
  int father_id = root->get_id();

  // since each node has at most three branches.
  Node* first_child = root->get_first_child();
  if (first_child) {
    int first_child_id = first_child->get_id();
    _left_branch[father_id] = first_child_id;

    if (first_child->get_next_sibling()) {
      first_child = first_child->get_next_sibling();
      int sec_child_id = first_child->get_id();
      _middle_branch[father_id] = sec_child_id;
    }

    if (first_child->get_next_sibling()) {
      Node* third_child = first_child->get_next_sibling();
      int third_child_id = third_child->get_id();
      _right_branch[father_id] = third_child_id;
    }
  }

  if (root->get_first_child()) {
    preOrderTraversal(root->get_first_child());
  }
  if (root->get_next_sibling()) {
    preOrderTraversal(root->get_next_sibling());
  }
}

/**
 * @brief Find all paths whose length is greater than the 'max_length'.
 */
std::vector<std::vector<int>> TreeBuild::findPathOverMaxLength(Node* root, int sum, int max_length, std::vector<std::vector<int>>& res)
{
  std::vector<int> vec;
  deepFirstSearch(root, sum, res, vec, max_length);
  return res;
}

void TreeBuild::deepFirstSearch(Node* root, int sum, std::vector<std::vector<int>>& res, std::vector<int> vec, int max_length)
{
  Node* first_child = root->get_first_child();
  int id = root->get_id();
  vec.push_back(id);

  if (first_child) {
    int dis1 = manhattanDistance(first_child, root);
    int sum1 = sum + dis1;
    deepFirstSearch(first_child, sum1, res, vec, max_length);
  } else {
    if (sum >= max_length) {
      res.push_back(vec);
    }
  }

  Node* child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
  }
  if (child) {
    int dis1 = manhattanDistance(root, root->get_father_node());
    int dis2 = manhattanDistance(child, child->get_father_node());  /////
    int sum2 = sum - dis1 + dis2;
    vec.pop_back();
    deepFirstSearch(child, sum2, res, vec, max_length);
  }
}

/**
 * @brief Returns the index of the endpoint of each segment and it's wire length.
 */
void TreeBuild::segmentIndexAndLength(Node* root, std::vector<std::pair<int, int>>& wire_segment_idx, std::vector<int>& length)
{
  int child_id = root->get_id();

  if (root->get_father_node()) {
    int fathe_id = root->get_father_node()->get_id();
    int length_dbu
        = abs(_points[child_id].get_x() - _points[fathe_id].get_x()) + abs(_points[child_id].get_y() - _points[fathe_id].get_y());

    // printf("ID: %d \t neirhbor:%d\n", fathe_id, child_id);
    wire_segment_idx.push_back(std::pair(fathe_id, child_id));

    length.push_back(length_dbu);
  }

  if (root->get_first_child()) {
    segmentIndexAndLength(root->get_first_child(), wire_segment_idx, length);
  }
  if (root->get_next_sibling()) {
    segmentIndexAndLength(root->get_next_sibling(), wire_segment_idx, length);
  }
}

void TreeBuild::convertToBinaryTree(Node* root)
{
  auto children = root->get_children();
  if (!root || children.empty())
    return;

  convertToBinaryTree(children[0]);

  if (children.size() == 2) {
    convertToBinaryTree(children[1]);
  } else if (children.size() > 2) {
    // 遍历其余的兄弟节点，并连接斯坦纳点
    auto current = root;
    auto p = root->get_location();
    for (size_t i = 1; i < children.size(); ++i) {
      if (children.size() - i == 2) {
        // 斯坦纳点
        auto steinerPoint = new Node(p);
        int steiner_id = _points.size();
        steinerPoint->set_id(steiner_id);
        _points.push_back(p);
        current->get_first_child()->set_next_sibling(steinerPoint);
        steinerPoint->set_father_node(current);

        // steinerPoint->set_first_child(children[i]);
        children[i]->set_father_node(steinerPoint);
        // steinerPoint->get_first_child()->set_next_sibling(children[i + 1]);
        children[i + 1]->set_father_node(steinerPoint);
        convertToBinaryTree(steinerPoint->get_first_child());
        convertToBinaryTree(steinerPoint->get_first_child()->get_next_sibling());
        break;
      } else {
        // 斯坦纳点
        p = children[i]->get_location();
        auto steinerPoint = new Node(p);
        int steiner_id = _points.size();
        steinerPoint->set_id(steiner_id);
        _points.push_back(p);
        // current->get_first_child()->set_next_sibling(steinerPoint);
        steinerPoint->set_father_node(current);
        // steinerPoint->set_first_child(children[i]);
        children[i]->set_father_node(steinerPoint);
        convertToBinaryTree(steinerPoint->get_first_child());
        current = steinerPoint;
      }
    }
  }
}
}  // namespace ito