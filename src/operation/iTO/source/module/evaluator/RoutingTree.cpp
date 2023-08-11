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
#include "RoutingTree.h"

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ito {
int RoutingTree::_null_pt = -1;

void getConnectedPins(Net *net,
                      // Return value.
                      DesignObjSeq &pins) {
  pins.clear();
  DesignObject *obj;
  FOREACH_NET_PIN(net, obj) { pins.push_back(obj); }
}

void RoutingTree::plotConectionInfo(const std::string file_name,
                                    const std::string net_name, RoutingTree *tree,
                                    Rectangle core) {
  std::string path = "/output/";

  std::ofstream dot_connection(path + file_name + ".gds");
  if (!dot_connection.good()) {
    std::cerr << "write_connection:: cannot open `" << file_name << "' for writing."
              << std::endl;
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
  std::vector<int>                 length_per_wire;

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
    feed << tree->get_location(index1).get_x() << " : "
         << tree->get_location(index1).get_y() << std::endl;
    feed << tree->get_location(index2).get_x() << " : "
         << tree->get_location(index2).get_y() << std::endl;
    feed << "ENDEL" << std::endl;
  }
  feed << "ENDSTR" << std::endl;
  feed << "ENDLIB" << std::endl;
  dot_connection << feed.str();
  feed.clear();
  dot_connection.close();
}

void RoutingTree::plotPinName(std::stringstream &feed, const std::string net_name,
                              RoutingTree *tree, int id) {
  feed << "TEXT" << std::endl;
  feed << "LAYER 1" << std::endl;
  feed << "TEXTTYPE 0" << std::endl;
  feed << "PRESENTATION 0,2,0" << std::endl;
  feed << "PATHTYPE 1" << std::endl;
  feed << "STRANS 0,0,0" << std::endl;
  feed << "MAG 1875" << std::endl;
  feed << "XY" << std::endl;
  auto loc = tree->get_location(id);
  int  pin_x = loc.get_x();
  int  pin_y = loc.get_y();
  feed << pin_x << " : " << pin_y << std::endl;

  if (tree->get_pin(id)) { // the real pin.
    const char *name = tree->get_pin(id)->get_name();
    string      str(name);
    feed << "STRING " + str << std::endl;
  } else { // the stainer pin.
    feed << "STRING " + net_name + " : " + std::to_string(id) << std::endl;
  }
  feed << "ENDEL" << std::endl;
}

RoutingTree *makeRoutingTree(Net *net, TimingDBAdapter *db_adapter,
                             RoutingType rout_type) {
  RoutingTree  *tree = new RoutingTree();
  DesignObjSeq &pins = tree->get_pins();
  vector<Point> points;
  // get drvr
  DesignObject *drvr_pin_port = net->getDriver();
  if (!drvr_pin_port) {
    delete tree;
    return nullptr;
  }

  TimingIDBAdapter       *idb_adapter = dynamic_cast<TimingIDBAdapter *>(db_adapter);
  IdbCoordinate<int32_t> *loc = idb_adapter->idbLocation(drvr_pin_port);
  Point                   drvr_loc = Point(loc->get_x(), loc->get_y());

  getConnectedPins(net, pins);
  int pin_count = pins.size();
  if (pin_count >= 2) {
    int *x = new int[pin_count];
    int *y = new int[pin_count];

    bool  have_set_root = false;
    Node *root = nullptr;

    Point before;
    int   pin_idx = 0;
    for (int i = 0; i < pin_count; i++) {
      DesignObject     *pin = pins[i];
      TimingIDBAdapter *idb_adapter = dynamic_cast<TimingIDBAdapter *>(db_adapter);

      IdbCoordinate<int32_t> *loc = idb_adapter->idbLocation(pin);
      Point                   pin_loc = Point(loc->get_x(), loc->get_y());

      points.push_back(pin_loc);

      // Designate drvr_pin_port as root
      if (drvr_loc == pin_loc) {
        Node *root_drvr = new Node(drvr_loc);
        tree->set_root(root_drvr);
        have_set_root = true;
      }

      x[pin_idx] = pin_loc.get_x();
      y[pin_idx] = pin_loc.get_y();
      pin_idx++;
    }
    if (!have_set_root) {
      root = new Node(Point(x[0], y[0]));
      tree->set_root(root);
    } else {
      delete root;
    }
    tree->set_points(points);
    switch (rout_type) {
    case RoutingType::kHVTree: tree->makeHVTree(x, y); break;
    case RoutingType::kSteiner: tree->makeSteinerTree(x, y); break;
    }
    // tree->makeSteinerTree(x, y);
    // tree->makeHVTree(x, y);
    // printTree(tree->get_root());
    // tree->updateBranch();
    delete[] x;
    delete[] y;
    return tree;
  }
  delete tree;
  return nullptr;
}

static int manhattanDistance(Node *n0, Node *n1) {
  Point p0 = n0->get_location();
  Point p1 = n1->get_location();
  int   x0 = p0.get_x();
  int   x1 = p1.get_x();

  int y0 = p0.get_y();
  int y1 = p1.get_y();

  return abs(x0 - x1) + abs(y0 - y1);
}

/**
 * @brief return max distance from driver pin to load pins.
 */
void findMaxDistfromDrvr(Node *root, int curr_dist, int &max_dist) {
  Node *first_child = root->get_first_child();

  if (first_child) {
    int dis = manhattanDistance(first_child, first_child->get_father_node());
    int sum = curr_dist + dis;
    findMaxDistfromDrvr(first_child, sum, max_dist);
  } else {
    max_dist = curr_dist > max_dist ? curr_dist : max_dist;
  }

  Node *child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
    int dis1 = manhattanDistance(child, child->get_father_node()); /////
    int dis = manhattanDistance(root, root->get_father_node());
    int sum2 = curr_dist - dis + dis1;
    findMaxDistfromDrvr(child, sum2, max_dist);
  }
}

/**
 * @brief length from driver pin to all load pins
 */
void RoutingTree::drvrToLoadLength(Node *root, vector<int> &load_index,
                                   vector<int> &length, int curr_dist) {
  Node *first_child = root->get_first_child();
  if (first_child) {
    int dis = manhattanDistance(first_child, first_child->get_father_node());
    int sum = curr_dist + dis;
    if (first_child->get_id() < (int)_pins.size()) {
      load_index.push_back(first_child->get_id());
      length.push_back(sum);
    }
    drvrToLoadLength(first_child, load_index, length, sum);
  }

  Node *child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
    int dis1 = manhattanDistance(child, child->get_father_node()); /////
    int dis = manhattanDistance(root, root->get_father_node());
    int sum = curr_dist - dis + dis1;
    if (child->get_id() < (int)_pins.size()) {
      load_index.push_back(child->get_id());
      length.push_back(sum);
    }
    drvrToLoadLength(child, load_index, length, sum);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void printTree(Node *root) {
  printf("Id:%-2d    loc:(%-5d, %-5d)    neighbor:%-2d", root->get_id(),
         root->get_location().get_x(), root->get_location().get_y(),
         root->get_neighbor());
  // father
  if (root->get_father_node()) {
    printf("\tfather:%-2d", root->get_father_node()->get_id());
  } else {
    printf("\tfather:%-2d", -1);
  }
  // first child
  if (root->get_first_child()) {
    printf("\tfirst_child:%-2d", root->get_first_child()->get_id());
  } else {
    printf("\tfirst_child:%-2d", -1);
  }
  // next sibling
  if (root->get_next_sibling()) {
    printf("\tnext_sibli:%-2d\n", root->get_next_sibling()->get_id());
  } else {
    printf("\tnext_sibli:%-2d\n", -1);
  }

  if (root->get_first_child()) {
    printTree(root->get_first_child());
  }
  if (root->get_next_sibling()) {
    printTree(root->get_next_sibling());
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief make RoutinTree by using FLUTE
 *
 */
void RoutingTree::makeSteinerTree(int x[], int y[]) {
  // steiner tree
  Flute::Tree stTree;
  // int flute_accuracy = 3;
  int pin_count = _points.size();

  stTree = Flute::flute(pin_count, x, y, FLUTE_ACCURACY);
  // Flute::printtree(stTree);
  // Find driver steiner point.
  XCord drvr_x_coordinate = _root->get_location().get_x();
  YCord drvr_y_coordinate = _root->get_location().get_y();

  int                 drvr_steiner_pt{0};
  int                 branch_count = stTree.deg * 2 - 2;
  bool                find_drvr = true;
  vector<vector<int>> adj(branch_count);
  for (int i = 0; i < branch_count; i++) {
    Flute::Branch &branch_pt = stTree.branch[i];
    int            j = branch_pt.n;
    // printf("(%d, %d)\n", i, j);
    if (i != j) {
      adj[i].push_back(j);
      adj[j].push_back(i);
    }
    if (find_drvr) {
      if (branch_pt.x == drvr_x_coordinate && branch_pt.y == drvr_y_coordinate) {
        drvr_steiner_pt = i;
        find_drvr = false;
      }
    }
  }

  // add steiner points
  for (int i = stTree.deg; i < branch_count; i++) {
    _points.push_back(Point(stTree.branch[i].x, stTree.branch[i].y));
  }

  int root_id = findIndexInPoints(_root);
  _root->set_id(root_id);

  findLeftRight(_root, -1, drvr_steiner_pt, adj, stTree);

  _is_visit_pin.resize(_points.size(), 0);
}

void RoutingTree::findLeftRight(Node *father, int father_id, int child_id,
                                vector<vector<int>> &adj, Flute::Tree stTree) {
  Node *child = nullptr;
  for (int j : adj[child_id]) {
    Point pt = Point(stTree.branch[j].x, stTree.branch[j].y);
    if (j != father_id) {
      child = new Node(pt);
      int id_child = j;
      insertTree(father, child, id_child);
      // cout << child_id << " , " << j << endl;
      findLeftRight(child, child_id, j, adj, stTree);
    }
  }
}

void RoutingTree::insertTree(Node *father, Node *child, int id_child) {
  Node *first_child = father->get_first_child();
  if (first_child == nullptr) {
    father->set_first_child(child);
    int neighbor = findIndexInPoints(father);
    child->set_neighbor(neighbor);
    child->set_father_node(father);

    int id = findIndexInPoints(child, id_child);
    child->set_id(id);

  } else {
    while (first_child->get_next_sibling()) {
      first_child = first_child->get_next_sibling();
    }
    first_child->set_next_sibling(child);
    child->set_neighbor(first_child->get_neighbor());

    child->set_father_node(father);

    int id = findIndexInPoints(child, id_child);
    child->set_id(id);
  }
}

int RoutingTree::findIndexInPoints(Node *n, int id) {
  int pin_size = _pins.size();
  if (id < pin_size) {
    return findIndexInPoints(n);
  } else {
    if (n->get_id() != -1) {
      return n->get_id();
    }
    auto target_pt = n->get_location();
    int  num_of_pt = _points.size();
    for (int i = _pins.size(); i < num_of_pt; ++i) {
      if (target_pt == _points[i] && !_points[i].isVisit()) {
        n->set_id(i);
        _points[i].set_is_visit();
        return i;
      }
    }
  }
  cout << "Can't find target pt in Points" << endl;
  exit(1);
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////

void RoutingTree::makeHVTree(int x[], int y[]) {
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

  int pin_count = _points.size();
  assert(pin_count == (int)_pins.size());

  std::multimap<YCord, XCord> y_to_x_map;
  // Determine how many intermediate nodes should be inserted.
  std::set<YCord> coordinate_set;
  for (int i = 0; i != pin_count; ++i) {
    coordinate_set.insert(y[i]);

    pair<YCord, XCord> tmp(y[i], x[i]);
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
    _points.push_back(Point(x_gravity, coordinate)); // add steiner point behind "_points"
    y_cordinate_vec.push_back(coordinate);
  }

  XCord drvr_x_coordinate = _root->get_location().get_x();
  YCord drvr_y_coordinate = _root->get_location().get_y();

  int pin_count_curr_y = y_to_x_map.count(drvr_y_coordinate);
  std::multimap<int, int>::iterator iter_curr;
  iter_curr = y_to_x_map.find(drvr_y_coordinate);

  // "x_gravity" is exactly the x-coordinate of a pin
  bool          x_gravity_at_pin = false;
  vector<XCord> pin_x_cord;
  for (int i = 0; i < pin_count_curr_y; i++, ++iter_curr) {
    pin_x_cord.push_back(iter_curr->second);
    if (x_gravity == iter_curr->second) {
      x_gravity_at_pin = true;
    }
  }
  if (!x_gravity_at_pin) {
    pin_x_cord.push_back(x_gravity);
    sort(pin_x_cord.begin(), pin_x_cord.end());
    pin_count_curr_y += 1;
  }

  // find the index of drvr pin
  vector<XCord>::iterator itr =
      find(pin_x_cord.begin(), pin_x_cord.end(), drvr_x_coordinate);
  int   drvr_idx = distance(pin_x_cord.begin(), itr);
  int   idx_begin_right = drvr_idx + 1;
  int   idx_begin_left = drvr_idx - 1;
  Node *first_steiner_pt = nullptr;

  // go right
  Node *parent1 = _root;
  while (idx_begin_right < pin_count_curr_y) {
    Point p_child = Point(pin_x_cord[idx_begin_right], drvr_y_coordinate);
    Node *child = new Node(p_child);
    insertTree(parent1, child);
    if (pin_x_cord[idx_begin_right] == x_gravity
        // steiner pt not at drvr pin location
        && x_gravity != drvr_x_coordinate) {
      first_steiner_pt = child;
    }
    parent1 = child;
    idx_begin_right++;
  }
  // go left
  Node *parent2 = _root;
  while (idx_begin_left >= 0) {
    Point p_child2 = Point(pin_x_cord[idx_begin_left], drvr_y_coordinate);
    Node *child2 = new Node(p_child2);
    insertTree(parent2, child2);
    if (pin_x_cord[idx_begin_left] == x_gravity && x_gravity != drvr_x_coordinate &&
        first_steiner_pt == nullptr) {
      first_steiner_pt = child2;
    }
    parent2 = child2;
    idx_begin_left--;
  }

  if (first_steiner_pt != nullptr) {
    int neighbor = findIndexInPoints(first_steiner_pt);
    int id = findIndexInPoints(_root);
    // first_steiner_pt->set_neighbor(neighbor);
    _root->set_neighbor(neighbor);
    _root->set_id(id);
  } else {
    first_steiner_pt = _root;
    int id = findIndexInPoints(_root);
    _root->set_neighbor(id);
    _root->set_id(id);
  }

  vector<int>::iterator it =
      find(y_cordinate_vec.begin(), y_cordinate_vec.end(), drvr_y_coordinate);
  auto index = std::distance(std::begin(y_cordinate_vec), it);
  int  index2 = index;

  ///////////////////////////////////////////////////
  Node *temp_node = first_steiner_pt;
  while (index > 0) {
    index--;
    updateTree(index, x_gravity, temp_node, y_cordinate_vec, y_to_x_map);
  }
  Node *temp_node2 = first_steiner_pt;
  int   number_of_intermediate_points = y_cordinate_vec.size() - 1;
  while (index2 < number_of_intermediate_points) {
    index2++;
    updateTree(index2, x_gravity, temp_node2, y_cordinate_vec, y_to_x_map);
  }

  _is_visit_pin.resize(_points.size(), 0);
}

/**
 * @brief Insert a child node of the parent node, and update node's info
 *
 * @param father
 * @param child
 */
void RoutingTree::insertTree(Node *father, Node *child) {
  Node *first_child = father->get_first_child();
  if (first_child == nullptr) {
    father->set_first_child(child);
    int neighbor = findIndexInPoints(father);
    child->set_neighbor(neighbor);
    child->set_father_node(father);

    int id = findIndexInPoints(child);
    child->set_id(id);

  } else {
    while (first_child->get_next_sibling()) {
      first_child = first_child->get_next_sibling();
    }
    first_child->set_next_sibling(child);
    child->set_neighbor(first_child->get_neighbor());

    child->set_father_node(father);

    int id = findIndexInPoints(child);
    child->set_id(id);
  }
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
void RoutingTree::updateTree(int index, XCord x_coordinate, Node *&parent,
                             std::vector<int>            cordinate_vec,
                             std::multimap<YCord, XCord> cordinate_to_x_map) {
  YCord cur_y = cordinate_vec[index];
  Node *father = new Node(Point(x_coordinate, cur_y));
  insertTree(parent, father);
  parent = father;

  int                               pin_count_curr_y = cordinate_to_x_map.count(cur_y);
  std::multimap<int, int>::iterator iter_curr;
  iter_curr = cordinate_to_x_map.find(cur_y);

  bool excute_once = true;
  // "x_gravity" is exactly the x-coordinate of a pin
  bool          x_gravity_at_pin = false;
  int           idx_begin_right = pin_count_curr_y - 1;
  vector<XCord> pin_x_cord;
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
  Node *parent2 = father;
  // go right
  while (idx_begin_right < pin_count_curr_y) {
    Point p_child = Point(pin_x_cord[idx_begin_right], cur_y);
    Node *child = new Node(p_child);
    insertTree(father, child);

    father = child;
    idx_begin_right++;
  }
  // go left
  while (idx_begin_left >= 0) {
    Point p_child = Point(pin_x_cord[idx_begin_left], cur_y);
    Node *child = new Node(p_child);
    insertTree(parent2, child);

    parent2 = child;
    idx_begin_left--;
  }
}

DesignObject *RoutingTree::get_pin(int idx) const {
  int pin_count = _pins.size();
  if (idx < pin_count) {
    return _pins[idx];
  } else {
    return nullptr;
  }
}

/**
 * @brief Return the index of node in the member "_points"
 */
int RoutingTree::findIndexInPoints(Node *n) {
  if (n->get_id() != -1) {
    return n->get_id();
  }
  auto target_pt = n->get_location();
  int  num_of_pt = _points.size();
  for (int i = 0; i < num_of_pt; ++i) {
    if (target_pt == _points[i] && !_points[i].isVisit()) {
      n->set_id(i);
      _points[i].set_is_visit();
      return i;
    }
  }
  cout << "Can't find target pt in Points" << endl;
  exit(1);
  return 0;
}

/**
 * @brief Update the branches(left/middle/right) of each node.
 *
 * This function needs to be executed before using each branch of the node
 */
void RoutingTree::updateBranch() {
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
void RoutingTree::preOrderTraversal(Node *root) {
  int father_id = root->get_id();

  // since each node has at most three branches.
  Node *first_child = root->get_first_child();
  if (first_child) {
    int first_child_id = first_child->get_id();
    _left_branch[father_id] = first_child_id;

    if (first_child->get_next_sibling()) {
      first_child = first_child->get_next_sibling();
      int sec_child_id = first_child->get_id();
      _middle_branch[father_id] = sec_child_id;
    }

    if (first_child->get_next_sibling()) {
      Node *third_child = first_child->get_next_sibling();
      int   third_child_id = third_child->get_id();
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
vector<vector<int>> RoutingTree::findPathOverMaxLength(Node *root, int sum,
                                                       int max_length,
                                                       // return values
                                                       vector<vector<int>> &res) {
  vector<int> vec;
  deepFirstSearch(root, sum, res, vec, max_length);
  return res;
}

void RoutingTree::deepFirstSearch(Node *root, int sum, vector<vector<int>> &res,
                                  vector<int> vec, int max_length) {
  Node *first_child = root->get_first_child();
  int   id = root->get_id();
  vec.push_back(id);

  if (first_child) {
    int dis1 = manhattanDistance(first_child, root);
    int sum1 = sum + dis1;
    deepFirstSearch(first_child, sum1, res, vec, max_length);
  } else {
    if (sum >= max_length) {
      res.push_back(vec);
      // printf("sum:%d\n", sum);
      // for (std::vector<int>::size_type i = 0; i < vec.size(); i++) {
      //   cout << vec[i] << ' ';
      // }
      // cout << endl;
    }
  }

  Node *child = nullptr;
  if (root->get_next_sibling()) {
    child = root->get_next_sibling();
  }
  if (child) {
    int dis1 = manhattanDistance(root, root->get_father_node());
    int dis2 = manhattanDistance(child, child->get_father_node()); /////
    int sum2 = sum - dis1 + dis2;
    vec.pop_back();
    deepFirstSearch(child, sum2, res, vec, max_length);
  }
}

/**
 * @brief Returns the index of the endpoint of each segment and it's wire length.
 */
void RoutingTree::segmentIndexAndLength(Node *root,
                                        // return values
                                        vector<pair<int, int>> &wire_segment_idx,
                                        vector<int>            &length) {
  // the point at pts[index1] and pts[index2] are connected.
  int child_id = root->get_id();

  if (root->get_father_node()) {
    int fathe_id = root->get_father_node()->get_id();
    int length_dbu = abs(_points[child_id].get_x() - _points[fathe_id].get_x()) +
                     abs(_points[child_id].get_y() - _points[fathe_id].get_y());

    // printf("ID: %d \t neirhbor:%d\n", fathe_id, child_id);
    wire_segment_idx.push_back(pair(fathe_id, child_id));

    length.push_back(length_dbu);
  }

  if (root->get_first_child()) {
    segmentIndexAndLength(root->get_first_child(), wire_segment_idx, length);
  }
  if (root->get_next_sibling()) {
    segmentIndexAndLength(root->get_next_sibling(), wire_segment_idx, length);
  }
}
} // namespace ito