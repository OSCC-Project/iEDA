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
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>

#include "lm_node.h"

namespace ilm {

// 为 std::unordered_map 定义哈希函数
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return static_cast<size_t>(p.first) * 16777619u ^ static_cast<size_t>(p.second);
    }
};

// 为 std::unordered_map 定义相等比较函数
struct PairEqual {
    bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};


class LmLayerGrid
{
 public:
  LmLayerGrid();
  ~LmLayerGrid();
    // 添加移动构造函数
  LmLayerGrid(LmLayerGrid&& other) noexcept
      : layer_order(other.layer_order),
        _node_map(std::move(other._node_map)),
        _map_mutex(std::move(other._map_mutex)),
        _row_num(other._row_num),
        _col_num(other._col_num)
  {
  }

  // 添加移动赋值运算符
  LmLayerGrid& operator=(LmLayerGrid&& other) noexcept {
      if (this != &other) {
          layer_order = other.layer_order;
          _node_map = std::move(other._node_map);
          _map_mutex = std::move(other._map_mutex);
          _row_num = other._row_num;
          _col_num = other._col_num;
      }
      return *this;
  }

  // getter
  LmNode* get_node(int row_id, int col_id, bool b_create = false);
  std::vector<LmNode*> get_all_nodes();
  // setter

  // operator
  std::pair<int, int> buildNodeMatrix(int order);
  LmNode* findNode(int x, int y);

 public:
  int layer_order;

 private:
  // 使用普通的 unordered_map 配合 shared_mutex
  std::unordered_map<std::pair<int, int>, LmNode*, PairHash, PairEqual> _node_map;
  std::unique_ptr<std::shared_mutex> _map_mutex; // 使用指针而不是直接成员
  int _row_num;
  int _col_num;
};

}  // namespace ilm
