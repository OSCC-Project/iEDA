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
/*
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-11-16 17:05:59
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-11-17 23:25:50
 * @FilePath: /irefactor/src/operation/iPL/source/utility/MultiTree.hh
 */

#ifndef IPL_MULTI_TREE_H
#define IPL_MULTI_TREE_H

#include <string>
#include <queue>

#include "data/Point.hh"
#include "module/topology_manager/TopologyManager.hh"

namespace ipl {

class TreeNode{
 public:
  TreeNode() = delete;
  explicit TreeNode(Point<int32_t> point);
  TreeNode(const TreeNode&) = delete;
  TreeNode(TreeNode&&) = delete;
  ~TreeNode() = default;

  TreeNode& operator=(const TreeNode&) = delete;
  TreeNode& operator=(TreeNode&&) = delete;

  // getter.
  Point<int32_t> get_point() const {return _point;}
  Node* get_node() const { return _node;}
  TreeNode* get_parent() const { return _parent;}
  std::vector<TreeNode*>& get_child_list() { return _child_list;}

  // setter.
  void set_node(Node* node) { _node = node;}
  void set_parent(TreeNode* parent) { _parent = parent;}
  void add_child(TreeNode* child) { _child_list.push_back(child);}

 private:
  Point<int32_t> _point;
  Node* _node;
  TreeNode* _parent;
  std::vector<TreeNode*> _child_list;
};
inline TreeNode::TreeNode(Point<int32_t> point) : _point(point), _node(nullptr), _parent(nullptr){}

class MultiTree{
  public:
    MultiTree() = delete;
    explicit MultiTree(TreeNode* root);
    MultiTree(const MultiTree&) = default;
    MultiTree(MultiTree&&) = default;
    ~MultiTree();

    MultiTree& operator=(const MultiTree&) = delete;
    MultiTree& operator=(MultiTree&&) = delete;

    // getter.
    TreeNode* get_root() const { return _root;}
    NetWork* get_network() const { return _network;}

    // setter.
    void set_network(NetWork* network) { _network = network;}

  private:
    TreeNode* _root;
    NetWork* _network;
};
inline MultiTree::MultiTree(TreeNode* root) : _root(root), _network(nullptr){}

inline MultiTree::~MultiTree(){
  std::queue<TreeNode*> node_queue;
  node_queue.push(_root);

  while(!node_queue.empty()){
    auto* parent = node_queue.front();
    for(auto* child : parent->get_child_list()){
      node_queue.push(child);
    }
    node_queue.pop();
    delete parent;
  }
}


}



#endif