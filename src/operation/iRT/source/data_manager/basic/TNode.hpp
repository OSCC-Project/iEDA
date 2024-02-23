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

namespace irt {

template <typename T>
class TNode
{
 public:
  TNode() = default;
  explicit TNode(const T& v) { _v = v; }
  ~TNode() = default;
  // getter
  T& value() { return _v; }
  std::vector<TNode<T>*>& get_child_list() { return _child_list; }
  // setter
  void set_value(const T& v) { _v = v; }
  // function
  int32_t getChildrenNum() { return static_cast<int32_t>(_child_list.size()); }
  bool isLeafNode() { return getChildrenNum() == 0; }

  void addChild(TNode<T>* child) { _child_list.push_back(child); }
  void addChildren(const std::vector<TNode<T>*>& child_list)
  {
    for (size_t i = 0; i < child_list.size(); i++) {
      addChild(child_list[i]);
    }
  }
  void delChild(TNode<T>* child) { _child_list.erase(find(_child_list.begin(), _child_list.end(), child)); }
  void delChildren(const std::vector<TNode<T>*>& child_list)
  {
    for (size_t i = 0; i < child_list.size(); i++) {
      delChild(child_list[i]);
    }
  }
  void clearChildren() { _child_list.clear(); }

 private:
  T _v;
  std::vector<TNode<T>*> _child_list;
};

}  // namespace irt
