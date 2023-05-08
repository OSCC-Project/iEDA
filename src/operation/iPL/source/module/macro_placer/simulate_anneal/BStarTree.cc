#include "BStarTree.hh"

#include <math.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace ipl::imp {

void BStarTree::inittree()
{
  _tree[0]->_parent = _num_macro;
  _tree[_num_macro]->_left = 0;
  int index = 1;
  for (int i = 0; i < _num_macro; ++i) {
    if (index == _num_macro) {
      continue;
    }
    _tree[i]->_left = index;
    _tree[index]->_parent = i;
    ++index;
    if (index == _num_macro) {
      continue;
    }
    _tree[i]->_right = index;
    _tree[index]->_parent = i;
    ++index;
  }
  for (size_t i = 0; i < _tree.size(); ++i) {
    _old_tree[i]->_parent = _tree[i]->_parent;
    _old_tree[i]->_left = _tree[i]->_left;
    _old_tree[i]->_right = _tree[i]->_right;
  }
  pack();
}

void BStarTree::rollback()
{
  if (!_rotate) {
    for (size_t i = 0; i < _tree.size(); ++i) {
      _tree[i]->_parent = _old_tree[i]->_parent;
      _tree[i]->_left = _old_tree[i]->_left;
      _tree[i]->_right = _old_tree[i]->_right;
    }
  } else {
    _macro_list[_rotate_macro_index]->set_orient(_old_orient);
  }
}

void BStarTree::update()
{
  if (!_rotate) {
    for (size_t i = 0; i < _tree.size(); ++i) {
      _old_tree[i]->_parent = _tree[i]->_parent;
      _old_tree[i]->_left = _tree[i]->_left;
      _old_tree[i]->_right = _tree[i]->_right;
    }
  }
}

void BStarTree::perturb()
{
  float rand_num = rand() / (RAND_MAX + 1.0);
  // swap node
  if (rand_num < _swap_pro) {
    _rotate = false;
    int block_a = int(_num_macro * (rand() / (RAND_MAX + 1.0)));
    int block_b = int((_num_macro - 1) * (rand() / (RAND_MAX + 1.0)));
    block_b = (block_b >= block_a) ? block_b + 1 : block_b;
    swap(block_a, block_b);
    // rotate node
  } else if (rand_num < 1 - _move_pro) {
    _rotate = true;
    _rotate_macro_index = int(_num_macro * (rand() / (RAND_MAX + 1.0)));
    _old_orient = _macro_list[_rotate_macro_index]->get_orient();
    int new_orient = (_rotate_macro_index + 1) % 8;  // the total of orient is 8
    _macro_list[_rotate_macro_index]->set_orient(Orient(new_orient));
    // move node
  } else {
    _rotate = false;
    int block = int(_num_macro * (rand() / (RAND_MAX + 1.0)));
    int target_rand_num = int((2 * _num_macro - 1) * (rand() / (RAND_MAX + 1.0)));
    int target = target_rand_num / 2;
    target = (target == block) ? target - 1 : target;
    target = (target < 0) ? 1 : target;
    int left_child = target_rand_num % 2;
    move(block, target, (left_child != 0));
  }
  pack();
}

void BStarTree::swap(int index_one, int index_two)
{
  int index_one_left = _tree[index_one]->_left;
  int index_one_right = _tree[index_one]->_right;
  int index_one_parent = _tree[index_one]->_parent;

  int index_two_left = _tree[index_two]->_left;
  int index_two_right = _tree[index_two]->_right;
  int index_two_parent = _tree[index_two]->_parent;

  if (index_one == index_two_parent)
    swapParentChild(index_one, (index_two == _tree[index_one]->_left));
  else if (index_two == index_one_parent)
    swapParentChild(index_two, (index_one == _tree[index_two]->_left));
  else {
    // update around index_one
    _tree[index_one]->_parent = index_two_parent;
    _tree[index_one]->_left = index_two_left;
    _tree[index_one]->_right = index_two_right;

    if (index_one == _tree[index_one_parent]->_left)
      _tree[index_one_parent]->_left = index_two;
    else
      _tree[index_one_parent]->_right = index_two;

    if (index_one_left != _undefined)
      _tree[index_one_left]->_parent = index_two;

    if (index_one_right != _undefined)
      _tree[index_one_right]->_parent = index_two;

    // update around index_two
    _tree[index_two]->_parent = index_one_parent;
    _tree[index_two]->_left = index_one_left;
    _tree[index_two]->_right = index_one_right;

    if (index_two == _tree[index_two_parent]->_right)  // prevent from that two indexs have togeter parent->
      _tree[index_two_parent]->_right = index_one;     // that will resulting in their parent's child don't change->
    else
      _tree[index_two_parent]->_left = index_one;

    if (index_two_left != _undefined)
      _tree[index_two_left]->_parent = index_one;

    if (index_two_right != _undefined)
      _tree[index_two_right]->_parent = index_one;
  }
}
void BStarTree::swapParentChild(int _parent, bool is_left)
{
  int _parent_parent = _tree[_parent]->_parent;
  int _parent_left = _tree[_parent]->_left;
  int _parent_right = _tree[_parent]->_right;

  int child = (is_left) ? _tree[_parent]->_left : _tree[_parent]->_right;
  int child_left = _tree[child]->_left;
  int child_right = _tree[child]->_right;

  if (is_left) {
    _tree[_parent]->_parent = child;
    _tree[_parent]->_left = child_left;
    _tree[_parent]->_right = child_right;

    if (_parent == _tree[_parent_parent]->_left)
      _tree[_parent_parent]->_left = child;
    else
      _tree[_parent_parent]->_right = child;

    if (_parent_right != _undefined)
      _tree[_parent_right]->_parent = child;

    _tree[child]->_parent = _parent_parent;
    _tree[child]->_left = _parent;
    _tree[child]->_right = _parent_right;

    if (child_left != _undefined)
      _tree[child_left]->_parent = _parent;

    if (child_right != _undefined)
      _tree[child_right]->_parent = _parent;
  } else {
    _tree[_parent]->_parent = child;
    _tree[_parent]->_left = child_left;
    _tree[_parent]->_right = child_right;

    if (_parent == _tree[_parent_parent]->_left)
      _tree[_parent_parent]->_left = child;
    else
      _tree[_parent_parent]->_right = child;

    if (_parent_left != _undefined)
      _tree[_parent_left]->_parent = child;

    _tree[child]->_parent = _parent_parent;
    _tree[child]->_left = _parent_left;
    _tree[child]->_right = _parent;

    if (child_left != _undefined)
      _tree[child_left]->_parent = _parent;

    if (child_right != _undefined)
      _tree[child_right]->_parent = _parent;
  }
}

void BStarTree::move(int index, int target, bool _left_child)
{
  int index_parent = _tree[index]->_parent;
  int index_left = _tree[index]->_left;
  int index_right = _tree[index]->_right;

  // remove "index" from the tree
  if ((index_left != _undefined) && (index_right != _undefined))
    removeUpChild(index);
  else if (index_left != _undefined) {
    _tree[index_left]->_parent = index_parent;
    if (index == _tree[index_parent]->_left)
      _tree[index_parent]->_left = index_left;
    else
      _tree[index_parent]->_right = index_left;
  } else if (index_right != _undefined) {
    _tree[index_right]->_parent = index_parent;
    if (index == _tree[index_parent]->_left)
      _tree[index_parent]->_left = index_right;
    else
      _tree[index_parent]->_right = index_right;
  } else {
    if (index == _tree[index_parent]->_left)
      _tree[index_parent]->_left = _undefined;
    else
      _tree[index_parent]->_right = _undefined;
  }

  int target_left = _tree[target]->_left;
  int target_right = _tree[target]->_right;

  // add "index" to the required coordinate
  if (_left_child) {
    _tree[target]->_left = index;
    if (target_left != _undefined)
      _tree[target_left]->_parent = index;

    _tree[index]->_parent = target;
    _tree[index]->_left = target_left;
    _tree[index]->_right = _undefined;
  } else {
    _tree[target]->_right = index;
    if (target_right != _undefined)
      _tree[target_right]->_parent = index;

    _tree[index]->_parent = target;
    _tree[index]->_left = _undefined;
    _tree[index]->_right = target_right;
  }
}
void BStarTree::removeUpChild(int index)
{
  int index_parent = _tree[index]->_parent;
  int index_left = _tree[index]->_left;
  int index_right = _tree[index]->_right;

  _tree[index_left]->_parent = index_parent;
  if (index == _tree[index_parent]->_left)
    _tree[index_parent]->_left = index_left;
  else
    _tree[index_parent]->_right = index_left;

  int ptr = index_left;
  while (_tree[ptr]->_right != _undefined)
    ptr = _tree[ptr]->_right;

  _tree[ptr]->_right = index_right;
  _tree[index_right]->_parent = ptr;
}

void BStarTree::pack()
{
  clean_contour(_contour);
  // x- and y- shifts for new block to avoid obstacles
  new_block_x_shift = 0;
  new_block_y_shift = 0;

  int tree_prev = _num_macro;
  int tree_curr = _tree[_num_macro]->_left;  // start with first block

  while (tree_curr != _num_macro)  // until reach the root again
  {
    if (tree_prev == _tree[tree_curr]->_parent) {
      unsigned obstacle_id = UINT_MAX;
      int32_t obstacle_x_min, obstacle_x_max, obstacle_y_min, obstacle_y_max;
      int32_t new_x_min, new_x_max, new_y_min, new_y_max;

      if (isIntersectsObstacle(tree_curr, obstacle_id, new_x_min, new_x_max, new_y_min, new_y_max, obstacle_x_min, obstacle_x_max, obstacle_y_min,
                               obstacle_y_max)) {
        // 'add obstacle' and then resume building the tree from here
        int tree_parent = _tree[tree_curr]->_parent;
        int32_t block_height = new_y_max - new_y_min;
        int32_t block_width = new_x_max - new_x_min;

        if ((tree_curr == _tree[tree_parent]->_left && obstacle_y_max + block_height > _contour[tree_parent]->_ctl + (block_height * 0.5)
             && (obstacle_x_max + block_width) < _obstacleframe[0])
            || (tree_curr == _tree[tree_parent]->_right && (obstacle_y_max + block_height) > _obstacleframe[1])) {
          // _left child & shifting up makes it too high
          // or _right child & shifting up makes it too high;
          // shift the starting location of the block _right in x
          new_block_x_shift += obstacle_x_max - new_x_min;
          new_block_y_shift = 0;
        } else {
          // shift the block in y
          new_block_y_shift += obstacle_y_max - new_y_min;
        }
      } else {
        addContourBlock(tree_curr);

        // reset x- y- obstacle shift
        new_block_x_shift = 0;
        new_block_y_shift = 0;

        tree_prev = tree_curr;
        if (_tree[tree_curr]->_left != _undefined)
          tree_curr = _tree[tree_curr]->_left;
        else if (_tree[tree_curr]->_right != _undefined)
          tree_curr = _tree[tree_curr]->_right;
        else
          tree_curr = _tree[tree_curr]->_parent;
      }
    } else if (tree_prev == _tree[tree_curr]->_left) {
      tree_prev = tree_curr;
      if (_tree[tree_curr]->_right != _undefined)
        tree_curr = _tree[tree_curr]->_right;
      else
        tree_curr = _tree[tree_curr]->_parent;
    } else {
      tree_prev = tree_curr;
      tree_curr = _tree[tree_curr]->_parent;
    }
  }
  _total_width = _contour[_num_macro + 1]->_begin;

  int contour_ptr = _contour[_num_macro]->_next;
  _total_height = 0;

  _total_contour_area = 0;
  while (contour_ptr != _num_macro + 1) {
    _total_height = std::max(_total_height, _contour[contour_ptr]->_ctl);
    // Calculate contour area
    _total_contour_area += (_contour[contour_ptr]->_end - _contour[contour_ptr]->_begin) * _contour[contour_ptr]->_ctl;
    // go to _next pointer
    contour_ptr = _contour[contour_ptr]->_next;
  }
  _total_area = float(_total_width) * float(_total_height);
}

void BStarTree::clean_contour(std::vector<ContourNode*>& old_contour)
{
  int vec_size = old_contour.size();
  int ledge = vec_size - 2;
  int bedge = vec_size - 1;

  old_contour[ledge]->_next = bedge;
  old_contour[ledge]->_prev = _undefined;
  old_contour[ledge]->_begin = 0;
  old_contour[ledge]->_end = 0;
  old_contour[ledge]->_ctl = _infty;

  old_contour[bedge]->_next = _undefined;
  old_contour[bedge]->_prev = ledge;
  old_contour[bedge]->_begin = 0;
  old_contour[bedge]->_end = _infty;
  old_contour[bedge]->_ctl = 0;

  // reset obstacles (so we consider all of them again)
  if (_seen_obstacles.size() != get_num_obstacles())
    _seen_obstacles.resize(get_num_obstacles());
  fill(_seen_obstacles.begin(), _seen_obstacles.end(), false);
}

bool BStarTree::isIntersectsObstacle(const int tree_ptr, unsigned& obstacle_id, int32_t& new_x_min, int32_t& new_x_max, int32_t& new_y_min,
                                     int32_t& new_y_max, int32_t& obstacle_x_min, int32_t& obstacle_x_max, int32_t& obstacle_y_min,
                                     int32_t& obstacle_y_max)
// check if adding this new block intersects an obstacle,return obstacle_id if it does
// TODO: smarter way of storing/searching obstacles->
// Currently complexity is O(foreach add_block * foreach unseen_obstacle)
{
  if (get_num_obstacles() == 0)
    return false;  // don't even bother

  obstacle_id = _undefined;
  int contour_ptr = _undefined;
  int contour_prev = _undefined;

  findBlockLocation(tree_ptr, new_x_min, new_y_min, contour_prev, contour_ptr);

  // int block = _tree[tree_ptr]->_block_index;
  // int theta = _tree[tree_ptr]->_orient;

  // get the rest of the bbox of new block, if the new block were added
  new_x_max = new_x_min + _macro_list[tree_ptr]->get_width();
  new_y_max = new_y_min + _macro_list[tree_ptr]->get_height();

  // check if adding this new block will create a contour that
  // intersects with an obstacle
  for (unsigned i = 0; i < get_num_obstacles(); i++) {
    obstacle_x_min = _obstacles[i]->get_x();
    obstacle_y_min = _obstacles[i]->get_y();
    obstacle_x_max = obstacle_x_min + _obstacles[i]->get_width();
    obstacle_y_max = obstacle_y_min + _obstacles[i]->get_height();

    if ((new_x_max <= obstacle_x_min) || (new_x_min >= obstacle_x_max) || (new_y_max <= obstacle_y_min) || (new_y_min >= obstacle_y_max))
      continue;
    obstacle_id = i;
    return true;
  }

  return false;
}

void BStarTree::addContourBlock(const int tree_ptr)
{
  int contour_ptr = _undefined;
  int contour_prev = _undefined;
  int32_t new_xloc, new_yloc;

  findBlockLocation(tree_ptr, new_xloc, new_yloc, contour_prev, contour_ptr);

  _macro_list[tree_ptr]->set_x(new_xloc);
  _macro_list[tree_ptr]->set_y(new_yloc);

  // int tree_parent = _tree[tree_ptr]->_parent;
  _contour[tree_ptr]->_begin = _macro_list[tree_ptr]->get_x();

  _contour[tree_ptr]->_end = _macro_list[tree_ptr]->get_x() + _macro_list[tree_ptr]->get_width();
  _contour[tree_ptr]->_ctl = _macro_list[tree_ptr]->get_y() + _macro_list[tree_ptr]->get_height();
  _contour[tree_ptr]->_next = contour_ptr;
  _contour[tree_ptr]->_prev = contour_prev;

  _contour[contour_ptr]->_prev = tree_ptr;
  _contour[contour_prev]->_next = tree_ptr;
  _contour[contour_ptr]->_begin = _macro_list[tree_ptr]->get_x() + _macro_list[tree_ptr]->get_width();
  _contour[tree_ptr]->_begin = max(_contour[contour_prev]->_end, _contour[tree_ptr]->_begin);
}

void BStarTree::findBlockLocation(const int tree_ptr, int32_t& out_x, int32_t& out_y, int& contour_prev, int& contour_ptr)
{
  int tree_parent = _tree[tree_ptr]->_parent;
  contour_prev = _undefined;
  contour_ptr = _undefined;

  // int     block = _tree[tree_ptr]->_block_index;
  int32_t new_block_contour_begin;
  if (tree_ptr == _tree[tree_parent]->_left) {
    // to the right of _parent, start x where _parent _ends
    new_block_contour_begin = _contour[tree_parent]->_end;
    // use block that's right of _parent's contour for y
    contour_ptr = _contour[tree_parent]->_next;
  } else {
    // above _parent, use _parent's x
    new_block_contour_begin = _contour[tree_parent]->_begin;
    // use parent's contour for y
    contour_ptr = tree_parent;
  }

  new_block_contour_begin += new_block_x_shift;  // considering obstacles
  contour_prev = _contour[contour_ptr]->_prev;   // _begins of cPtr/tPtr match

  int32_t new_block_contour_end = new_block_contour_begin + _macro_list[tree_ptr]->get_width();
  uint32_t max_ctl = _contour[contour_ptr]->_ctl;
  int32_t contour_ptr_end =  // �ж��Ƿ�Ϊ���ڵ�
      (contour_ptr == tree_ptr) ? new_block_contour_end : _contour[contour_ptr]->_end;

  while (contour_ptr_end <= new_block_contour_end + _tolerance) {
    max_ctl = max(max_ctl, _contour[contour_ptr]->_ctl);
    contour_ptr = _contour[contour_ptr]->_next;
    contour_ptr_end = (contour_ptr == tree_ptr) ? new_block_contour_end : _contour[contour_ptr]->_end;
  }

  // contour_prev location Update!!!!
  int32_t contour_ptr_begin = (contour_ptr == tree_ptr) ? new_block_contour_begin : _contour[contour_ptr]->_begin;

  if (contour_ptr_begin + _tolerance < new_block_contour_end)
    max_ctl = max(max_ctl, _contour[contour_ptr]->_ctl);

  // get location where new block sho1uld be added
  out_x = new_block_contour_begin;
  out_y = max_ctl;
}

}  // namespace ipl::imp