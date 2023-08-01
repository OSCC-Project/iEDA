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

#include <limits.h>
#include <math.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "MPSolution.hh"
#include "Setting.hh"
#include "database/FPInst.hh"
#include "database/FPRect.hh"

static const int UNDEFINED = -1;

namespace ipl::imp {

class BStarTreeNode
{
 public:
  int _parent = UNDEFINED;
  int _left = UNDEFINED;
  int _right = UNDEFINED;
};

class ContourNode
{
 public:
  int _next = 0;
  int _prev = 0;

  int32_t _begin = 0;
  int32_t _end = 0;
  uint32_t _ctl = 0;
};

class BStarTree : public MPSolution
{
 public:
  BStarTree(std::vector<FPInst*> macro_list, Setting* set);
  ~BStarTree();

  void inittree();
  void perturb() override;
  void pack() override;
  void rollback() override;
  void update() override;
  size_t get_num_obstacles() { return _obstacles.size(); }

  void clean_tree(std::vector<BStarTreeNode*>& old_tree);
  void swap(int index_one, int index_two);
  void move(int index, int target, bool left_child);
  void swapParentChild(int parent, bool is_left);
  void removeUpChild(int index);
  void clean_contour();

 private:
  bool isIntersectsObstacle(const int tree_ptr, unsigned& obstacle_id, int32_t& new_x_min, int32_t& new_x_max, int32_t& new_y_min,
                            int32_t& new_y_max, int32_t& obstacle_x_min, int32_t& obstacle_x_max, int32_t& obstacle_y_min,
                            int32_t& obstacle_y_max);
  void addContourBlock(int tree_ptr);
  void findBlockLocation(const int tree_ptr, int32_t& out_x, int32_t& out_y, int& contour_prev, int& contour_ptr);

  std::vector<BStarTreeNode*> _tree;
  std::vector<BStarTreeNode*> _old_tree;
  std::vector<ContourNode*> _contour;

  int32_t _tolerance;
  int32_t _new_block_x_shift;  // x shift for avoiding obstacles when adding new block
  int32_t _new_block_y_shift;  // y shift for avoiding obstacles when adding new block
  std::vector<FPRect*> _obstacles;
  int32_t _obstacleframe[2];          // 0=width, 1=height
  std::vector<bool> _seen_obstacles;  // track obstacles that have been consumed during pack

  float _total_contour_area;
  float _swap_pro;
  float _move_pro;
  bool _rotate;
  int _rotate_macro_index;
  Orient _old_orient;
};

}  // namespace ipl::imp