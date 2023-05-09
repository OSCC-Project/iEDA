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
 * @project		iplf
 * @file		file_cts.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "file_manager.h"

using std::string;
using std::unordered_map;
using std::vector;

namespace iplf {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CTS routing data structure
struct CtsFileNetHeader
{
  int32_t net_num;
};

struct CtsFileSegmentHeader
{
  char net_name[1000];
  int32_t segment_num;
};

struct CtsFileSegment
{
  char start_name[1000];
  int32_t start_x;
  int32_t start_y;
  char end_name[1000];
  int32_t end_x;
  int32_t end_y;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CTS GUI data structure
struct CtsFileTreeHeader
{
  int32_t tree_num;
};

struct CtsFileNodeHeader
{
  char root_name[1000];
  int64_t node_num;
};

struct CtsFileTreeNode
{
  char node_name[1000];
  char parent_name[1000];
  double delay;  /// delay between parent node and node
  int64_t leaf_num;
};

class CtsTreeNode
{
 public:
  explicit CtsTreeNode(CtsFileTreeNode node) { _node = node; }
  ~CtsTreeNode() = default;

  CtsFileTreeNode get_node() { return _node; }
  std::string get_node_name() { return _node.node_name; }
  std::string get_parent_name() { return _node.parent_name; }
  double get_delay() { return _node.delay; }
  int64_t get_leaf_num() { return _node.leaf_num; }

  void set_node(CtsFileTreeNode node) { _node = node; }
  void set_node_name(std::string name)
  {
    std::memset(_node.node_name, 0, 1000);
    std::memcpy(_node.node_name, name.c_str(), std::min(1000, (int) name.length()));
  }

  void set_parent_name(std::string name)
  {
    std::memset(_node.parent_name, 0, 1000);
    std::memcpy(_node.parent_name, name.c_str(), std::min(1000, (int) name.length()));
  }

  void set_delay(double delay) { _node.delay = delay; }
  void set_leaf_num(int num) { _node.leaf_num = num; }
  void add_leaf_num(int num) { _node.leaf_num += num; }

  bool is_root()
  {
    std::string name = _node.parent_name;
    return name.empty() ? true : false;
  }

  bool is_leaf() { return _child_node_list.size() > 0 ? false : true; }

  std::vector<CtsTreeNode*>& get_child_list() { return _child_node_list; }
  void addChildNode(CtsTreeNode* child_node) { _child_node_list.push_back(child_node); }
  void clear() { _child_node_list.clear(); }

  /// operation
  std::vector<CtsTreeNode*> get_child_leaf_list()
  {
    std::vector<CtsTreeNode*> leaf_list;
    for (auto node : _child_node_list) {
      if (node->is_leaf()) {
        leaf_list.push_back(node);
      }
    }

    return leaf_list;
  }

  std::vector<CtsTreeNode*> get_child_node_list()
  {
    std::vector<CtsTreeNode*> leaf_list;
    for (auto node : _child_node_list) {
      if (!node->is_leaf()) {
        leaf_list.push_back(node);
      }
    }

    return leaf_list;
  }

 private:
  CtsFileTreeNode _node;
  std::vector<CtsTreeNode*> _child_node_list;
};

class CtsTreeNodeMap
{
 public:
  CtsTreeNodeMap(int64_t size) { _node_map.reserve(size); }
  ~CtsTreeNodeMap() = default;

  int64_t get_buffer_size() { return _node_map.size() * sizeof(CtsFileTreeNode); }

  unordered_map<std::string, CtsTreeNode*>& get_node_map() { return _node_map; }

  void set_root(CtsTreeNode* node) { _root = node; }
  CtsTreeNode* get_root() { return _root; }
  std::string get_root_name() { return _root != nullptr ? _root->get_node_name() : ""; }

  void addNode(CtsTreeNode* node)
  {
    if (node != nullptr) {
      std::string name = node->get_node().node_name;
      _node_map.insert(std::make_pair(name, node));
    }
  }
  CtsTreeNode* findNode(string node_name)
  {
    if (_node_map.find(node_name) == _node_map.end()) {
      std::cout << "[Error] Can not found the node = " << node_name << std::endl;
      return nullptr;
    }
    return _node_map[node_name];
  }

  void clear()
  {
    for (auto node : _node_map) {
      if (node.second != nullptr) {
        node.second->clear();
        delete node.second;
        node.second = nullptr;
      }
    }

    _node_map.clear();
  }

  void updateChildNode();

 private:
  CtsTreeNode* _root = nullptr;
  unordered_map<std::string, CtsTreeNode*> _node_map;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class FileCtsManager : public FileManager
{
 public:
  explicit FileCtsManager(string data_path) : FileManager(data_path) {}

  explicit FileCtsManager(string data_path, int32_t object_id) : FileManager(data_path, FileModuleId::kCTS, object_id) {}
  ~FileCtsManager() = default;

 private:
  /// file parser
  virtual bool parseFileData() override;

  /// file save
  virtual int32_t getBufferSize() override;
  virtual bool saveFileData() override;

  /// pa data

 private:
  const int max_num = 100000;
  const int max_size = max_num * sizeof(CtsFileTreeNode) + 100;

  int32_t getCtsRoutingBufferSize();
  bool saveCtsRoutingResult();
  bool parseCtsRoutingResult();

  int32_t getCtsTreeDataSize();
  bool saveCtsTreeData();
  bool parseCtsTreeData();

  void updateChildNode();
};

}  // namespace iplf
