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
/**
 * @project		iplf
 * @file		file_cts.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "file_cts.h"

#include <iostream>

#include "icts_io/icts_io.h"

using namespace std;

namespace iplf {

void CtsTreeNodeMap::updateChildNode()
{
  for (auto [node_name, node] : _node_map) {
    if (node == nullptr) {
      continue;
    }

    auto parent_node = findNode(node->get_parent_name());
    if (parent_node != nullptr) {
      parent_node->addChildNode(node);
    } else {
      std::string node_name = node->get_parent_name();
      std::cout << "[Error] : can not find parent node : " << node->get_parent_name() << std::endl;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FileCtsManager::getBufferSize()
{
  CtsDbId obj_id = (CtsDbId) (get_object_id());
  switch (obj_id) {
    case CtsDbId::kCtsRoutingData:
      return getCtsRoutingBufferSize();
      break;
    case CtsDbId::kCtsGuiData:
      return getCtsTreeDataSize();
      break;

    default:
      break;
  }
  return 0;
}

bool FileCtsManager::parseFileData()
{
  CtsDbId obj_id = (CtsDbId) (get_object_id());
  switch (obj_id) {
    case CtsDbId::kCtsRoutingData:
      return parseCtsRoutingResult();
      break;
    case CtsDbId::kCtsGuiData:
      return parseCtsTreeData();
      break;

    default:
      break;
  }
  return false;
}

bool FileCtsManager::saveFileData()
{
  CtsDbId obj_id = (CtsDbId) (get_object_id());
  switch (obj_id) {
    case CtsDbId::kCtsRoutingData:
      return saveCtsRoutingResult();
      break;
    case CtsDbId::kCtsGuiData:
      return saveCtsTreeData();
      break;

    default:
      break;
  }
  return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CTS Routing data
int32_t FileCtsManager::getCtsRoutingBufferSize()
{
  return ctsInst->get_routing_buffer_size();
}

bool FileCtsManager::saveCtsRoutingResult()
{
  int size = getBufferSize();
  //   assert(size != 0);

  char* data_buf = new char[size];
  std::memset(data_buf, 0, size);

  int mem_size = 0;
  char* buf_ref = data_buf;

  /// save cts data header
  CtsFileNetHeader cts_data_header;
  cts_data_header.net_num = ctsInst->get_net_size();
  std::memcpy(buf_ref, &cts_data_header, sizeof(CtsFileNetHeader));
  buf_ref += sizeof(CtsFileNetHeader);
  mem_size += sizeof(CtsFileNetHeader);

  for (auto [net_name, seg_list] : ctsInst->get_net_map()) {
    /// save segment header
    CtsFileSegmentHeader segment_header;
    std::memset(&segment_header, 0, sizeof(CtsFileSegmentHeader));
    std::memcpy(segment_header.net_name, net_name.c_str(), net_name.length());
    segment_header.segment_num = seg_list.size();
    std::memcpy(buf_ref, &segment_header, sizeof(CtsFileSegmentHeader));
    buf_ref += sizeof(CtsFileSegmentHeader);
    mem_size += sizeof(CtsFileSegmentHeader);

    for (auto seg : seg_list) {
      /// save unit segment
      std::memcpy(buf_ref, &seg, sizeof(CtsFileSegment));
      buf_ref += sizeof(CtsFileSegment);
      mem_size += sizeof(CtsFileSegment);
    }
  }

  if (size != mem_size) {
    std::cout << "Error : memory size error." << std::endl;
  }

  /// write file
  get_fstream().write(data_buf, size);

  get_fstream().seekp(size, ios::cur);

  delete[] data_buf;
  data_buf = nullptr;
  buf_ref = nullptr;

  return true;
}

bool FileCtsManager::parseCtsRoutingResult()
{
  int size = get_data_size();
  if (size == 0) {
    return false;
  }

  char* data_buf = new char[size];
  get_fstream().read(data_buf, size);
  get_fstream().seekp(size, ios::cur);
  char* buf_ref = data_buf;

  /// parse cts data header
  CtsFileNetHeader cts_data_header;
  std::memcpy(&cts_data_header, buf_ref, sizeof(CtsFileNetHeader));
  buf_ref += sizeof(CtsFileNetHeader);

  unordered_map<string, vector<CtsFileSegment>>& net_map = ctsInst->get_net_map();
  for (int i = 0; i < cts_data_header.net_num; ++i) {
    /// parse cts header
    CtsFileSegmentHeader cts_segment_header;
    std::memcpy(&cts_segment_header, buf_ref, sizeof(CtsFileSegmentHeader));
    buf_ref += sizeof(CtsFileSegmentHeader);

    vector<CtsFileSegment> cts_segment_list;
    /// parse segment
    for (int j = 0; j < cts_segment_header.segment_num; ++j) {
      /// parse single unit
      CtsFileSegment segment;
      std::memcpy(&segment, buf_ref, sizeof(CtsFileSegment));

      /// add to segment list
      cts_segment_list.push_back(segment);
      buf_ref += sizeof(CtsFileSegment);
    }

    net_map.insert(std::make_pair(string(cts_segment_header.net_name), cts_segment_list));
  }
  delete[] data_buf;
  data_buf = nullptr;
  buf_ref = nullptr;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CTS GUI data
int32_t FileCtsManager::getCtsTreeDataSize()
{
  return ctsInst->getCtsTreeBufferSize();
}

bool FileCtsManager::saveCtsTreeData()
{
  auto& tree_list = ctsInst->getTreeData();
  /// save cts data header
  CtsFileTreeHeader file_header;
  file_header.tree_num = tree_list.size();

  get_fstream().write((char*) &file_header, sizeof(CtsFileTreeHeader));
  get_fstream().seekp(sizeof(CtsFileTreeHeader), ios::cur);

  char* data_buf = new char[max_size];
  std::memset(data_buf, 0, max_size);

  for (auto& tree : tree_list) {
    /// save tree header
    CtsFileNodeHeader tree_header;
    std::memset(&tree_header, 0, sizeof(CtsFileNodeHeader));

    tree_header.node_num = tree->get_node_map().size();
    std::string root_name = tree->get_root_name();
    std::memcpy(tree_header.root_name, root_name.c_str(), root_name.length());

    get_fstream().write((char*) &tree_header, sizeof(CtsFileNodeHeader));
    get_fstream().seekp(sizeof(CtsFileNodeHeader), ios::cur);

    char* buf_ref = data_buf;
    int index = 0;
    uint64_t total_num = 0;
    for (auto tree_node : tree->get_node_map()) {
      /// wrap drc file struct

      CtsFileTreeNode file_node;
      if (tree_node.second == nullptr) {
        std::memset(&file_node, 0, sizeof(CtsFileTreeNode));
      } else {
        file_node = tree_node.second->get_node();
      }
      /// save drc data
      std::memcpy(buf_ref, &file_node, sizeof(CtsFileTreeNode));
      buf_ref += sizeof(CtsFileTreeNode);
      index++;
      total_num++;

      if (index == max_num || total_num >= tree->get_node_map().size()) {
        /// write file
        get_fstream().write(data_buf, sizeof(CtsFileTreeNode) * index);
        get_fstream().seekp(sizeof(CtsFileTreeNode) * index, ios::cur);

        /// reset
        std::memset(data_buf, 0, max_size);
        buf_ref = data_buf;
        index = 0;
      }
    }

    buf_ref = nullptr;
  }

  delete[] data_buf;
  data_buf = nullptr;

  return true;
}

bool FileCtsManager::parseCtsTreeData()
{
  //   int64_t size = getCtsTreeDataSize();
  //   if (size == 0) {
  //     return false;
  //   }

  /// parse cts data header
  CtsFileTreeHeader data_header;
  get_fstream().read((char*) &data_header, sizeof(CtsFileTreeHeader));
  get_fstream().seekp(sizeof(CtsFileTreeHeader), ios::cur);

  ctsInst->clear();

  char* data_buf = new char[max_size];
  std::memset(data_buf, 0, max_size);

  for (int i = 0; i < data_header.tree_num; ++i) {
    /// add tree
    CtsTreeNodeMap* node_map = ctsInst->addTreeNodeMap();

    /// parse header
    CtsFileNodeHeader tree_header;
    get_fstream().read((char*) &tree_header, sizeof(CtsFileNodeHeader));
    get_fstream().seekp(sizeof(CtsFileNodeHeader), ios::cur);

    std::memset(data_buf, 0, max_size);
    char* buf_ref = data_buf;
    decltype(tree_header.node_num) total_num = 0;

    while (total_num < tree_header.node_num) {
      /// calculate spot number read from file
      int read_num = tree_header.node_num - total_num >= max_num ? max_num : tree_header.node_num - total_num;

      get_fstream().read(data_buf, sizeof(CtsFileTreeNode) * read_num);
      get_fstream().seekp(sizeof(CtsFileTreeNode) * read_num, ios::cur);

      for (int j = 0; j < read_num; j++) {
        /// parse single unit
        CtsFileTreeNode file_node;
        std::memcpy(&file_node, buf_ref, sizeof(CtsFileTreeNode));
        CtsTreeNode* node = new CtsTreeNode(file_node);

        /// add to list
        node_map->addNode(node);

        buf_ref += sizeof(CtsFileTreeNode);

        total_num++;
      }

      std::memset(data_buf, 0, max_size);
      buf_ref = data_buf;
    }

    /// set root
    std::string root_name = tree_header.root_name;
    auto root_node = node_map->findNode(root_name);
    node_map->set_root(root_node);
  }

  delete[] data_buf;
  data_buf = nullptr;

  /// build tree
  updateChildNode();

  return true;
}

void FileCtsManager::updateChildNode()
{
  for (auto& node_map : ctsInst->get_node_list()) {
    node_map->updateChildNode();
  }
}

}  // namespace iplf
