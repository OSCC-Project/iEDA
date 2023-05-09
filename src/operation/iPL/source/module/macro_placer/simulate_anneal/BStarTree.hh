
#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "MPSolution.hh"
#include "Setting.hh"
#include "database/FPInst.hh"
#include "database/FPRect.hh"

// #include <limits>
static const int _undefined = -1;
static const uint32_t _infty = 2147483647;
static const int _default = 0;

using std::string;
using std::vector;

// using std::numeric_limits;
namespace ipl::imp {

class BStarTreeNode
{
 public:
  BStarTreeNode() {}
  ~BStarTreeNode() = default;

  int _parent = _undefined;
  int _left = _undefined;
  int _right = _undefined;
};

class ContourNode
{
 public:
  ContourNode() {}
  int _next = 0;
  int _prev = 0;

  int32_t _begin = 0;
  int32_t _end = 0;
  uint32_t _ctl = 0;
};

class BStarTree : public MPSolution
{
 public:
  BStarTree(vector<FPInst*> macro_list, Setting* set) : MPSolution(macro_list)
  {
    for (int i = 0; i < _num_macro + 2; ++i) {
      BStarTreeNode* tree_node = new BStarTreeNode();
      BStarTreeNode* old_tree_node = new BStarTreeNode();
      tree_node->_parent = _undefined;
      tree_node->_left = _undefined;
      tree_node->_right = _undefined;
      old_tree_node->_parent = _undefined;
      old_tree_node->_left = _undefined;
      old_tree_node->_right = _undefined;
      _tree.emplace_back(tree_node);
      _old_tree.emplace_back(old_tree_node);

      ContourNode* contour_node = new ContourNode();
      contour_node->_next = _undefined;
      contour_node->_prev = _undefined;
      contour_node->_begin = _undefined;
      contour_node->_ctl = _undefined;
      _contour.emplace_back(contour_node);
    }
    _contour[_num_macro]->_next = _num_macro + 1;
    _contour[_num_macro]->_prev = _undefined;
    _contour[_num_macro]->_begin = 0;
    _contour[_num_macro]->_end = 0;
    _contour[_num_macro]->_ctl = _infty;
    _contour[_num_macro + 1]->_next = _undefined;
    _contour[_num_macro + 1]->_prev = _num_macro;
    _contour[_num_macro + 1]->_begin = 0;
    _contour[_num_macro + 1]->_end = _infty;
    _contour[_num_macro + 1]->_ctl = 0;

    FPInst* inst1 = new FPInst();
    FPInst* inst2 = new FPInst();
    inst1->set_height(_infty);
    inst2->set_width(_infty);
    _macro_list.emplace_back(inst1);
    _macro_list.emplace_back(inst2);

    _swap_pro = set->get_swap_pro();
    _move_pro = set->get_move_pro();
    inittree();
  }

  ~BStarTree(){};

  enum MoveType
  {
    SWAP,
    ROTATE,
    MOVE
  };

  void inittree();
  void perturb() override;
  void pack() override;
  void rollback() override;
  void update() override;
  size_t get_num_obstacles() { return _obstacles.size(); }
  int32_t get_block_area() const { return _block_area; };
  int get_num_macro() const { return _num_macro; }

  void clean_tree(std::vector<BStarTreeNode*>& old_tree);
  void swap(int index_one, int index_two);
  void move(int index, int target, bool left_child);
  void swapParentChild(int parent, bool is_left);
  void removeUpChild(int index);
  void clean_contour(std::vector<ContourNode*>& old_contour);

  vector<BStarTreeNode*> _tree{};
  vector<BStarTreeNode*> _old_tree{};
  vector<ContourNode*> _contour{};

 private:
  bool isIntersectsObstacle(const int tree_ptr, unsigned& obstacleID, int32_t& new_xMin, int32_t& new_xMax, int32_t& new_yMin,
                            int32_t& new_yMax, int32_t& obstacle_xMin, int32_t& obstacle_xMax, int32_t& obstacle_yMin,
                            int32_t& obstacle_yMax);
  void addContourBlock(int treePtr);
  void findBlockLocation(const int tree_ptr, int32_t& out_x, int32_t& out_y, int&, int&);

  int32_t _tolerance = 0;

  int32_t new_block_x_shift = 0;  // x shift for avoiding obstacles when adding new block
  int32_t new_block_y_shift = 0;  // y shift for avoiding obstacles when adding new block
  vector<FPRect*> _obstacles;
  int32_t _obstacleframe[2];            // 0=width, 1=height
  std::vector<bool> _seen_obstacles{};  // track obstacles that have been consumed during pack

  int32_t _block_area = 0;
  float _total_contour_area = 0;
  float _outline_rito;

  float _swap_pro = 0.3;
  float _move_pro = 0.3;

  bool _rotate = false;
  int _rotate_macro_index = 0;
  Orient _old_orient = Orient::N;
};

}  // namespace ipl::imp