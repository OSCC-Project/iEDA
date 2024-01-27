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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "file_cts.h"

#define ctsInst (iplf::CtsIO::getInstance())

namespace ista {
class StaClockTree;
}

namespace iplf {

class CtsIO
{
 public:
  static CtsIO* getInstance()
  {
    if (!_instance) {
      _instance = new CtsIO;
    }
    return _instance;
  }

  /// getter

  /// io
  bool runCTS(std::string config = "", std::string work_dir = "");
  bool reportCTS(std::string path = "");
  // bool initTopo(std::string config = "");

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// CTS routing data manage
  unordered_map<string, vector<CtsFileSegment>>& get_net_map() { return _net_map; }
  int32_t get_net_size() { return _net_map.size(); }
  int32_t get_routing_buffer_size();
  bool readCtsDataFromFile(string file_path);
  bool saveCtsDataToFile(string file_path);

  /// operator
  void addNet(string net_name, vector<CtsFileSegment> segment_list) { _net_map.insert(std::make_pair(net_name, segment_list)); }
  vector<CtsFileSegment> findSegmentList(string net_name)
  {
    if (_net_map.find(net_name) == _net_map.end()) {
      std::cout << "[Error] Not found the clock net " << net_name << " topo!" << std::endl;
      //   assert(false);
    }
    return _net_map[net_name];
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// CTS GUI data manage
  vector<CtsTreeNodeMap*>& get_node_list() { return _node_list; }
  int64_t getCtsTreeBufferSize();

  bool readTreeDataFromFile(std::string path = "");
  bool saveTreeDataToFile(std::string path);

  vector<CtsTreeNodeMap*>& getTreeData(std::string path = "");
  CtsTreeNodeMap* addTreeNodeMap(int64_t size = 0)
  {
    CtsTreeNodeMap* node_map = new CtsTreeNodeMap(size);
    _node_list.push_back(node_map);
    return node_map;
  }

  void clear()
  {
    for (auto node_map : _node_list) {
      if (node_map != nullptr) {
        node_map->clear();
      }

      delete node_map;
      node_map = nullptr;
    }

    _node_list.clear();
    vector<CtsTreeNodeMap*>().swap(_node_list);
  }

  void updateLeafNumber();

 private:
  static CtsIO* _instance;
  unordered_map<string, vector<CtsFileSegment>> _net_map;

  vector<CtsTreeNodeMap*> _node_list;

  CtsIO() {}
  ~CtsIO() = default;

  //   void buildData(icts::CTS& cts);
  void wrapTree(std::vector<std::unique_ptr<ista::StaClockTree>>& sta_tree_list);

  int64_t updateTotalNumberForLeaf(CtsTreeNode* node);
};

}  // namespace iplf
