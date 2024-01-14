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
#include "icts_io.h"

#include "builder.h"
#include "dm_cts_config.h"
#include "flow_config.h"
#include "iCTS/api/CTSAPI.hh"
#include "ista_io.h"
#include "source/module/sta/StaClockTree.hh"
#include "usage/usage.hh"

namespace iplf {
CtsIO* CtsIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CtsIO::runCTS(std::string config, std::string work_dir)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_icts_path();
  }

  flowConfigInst->set_status_stage("iCTS - Clock Tree Synthesis");

  ieda::Stats stats;

  CTSAPIInst.init(config, work_dir);
  CTSAPIInst.runCTS();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool CtsIO::reportCTS(std::string path)
{
  if (path.empty()) {
    path = flowConfigInst->get_icts_path();
  }
  CTSAPIInst.report(path);
  return true;
}

/**
 * @Brief : build cts data
 * @param  cts
 */
// void CtsIO::buildData(icts::CTS& cts)
// {
//   /// build cts data
//   std::map<std::string, vector<icts::CtsSignalWire>> net_topo_map;
//   cts.getClockNets(net_topo_map);

//   for (auto [net_name, topo_list] : net_topo_map) {
//     vector<CtsFileSegment> cts_segment_list;
//     cts_segment_list.reserve(topo_list.size());
//     for (size_t i = 0; i < topo_list.size(); i++) {
//       icts::CtsSignalWire& topo = topo_list[i];
//       icts::Endpoint first_point = topo.get_first();
//       icts::Endpoint second_point = topo.get_second();

//       CtsFileSegment cts_segment;

//       std::strcpy(cts_segment.start_name, first_point._name.c_str());
//       cts_segment.start_x = first_point._point.x();
//       cts_segment.start_y = first_point._point.y();

//       std::strcpy(cts_segment.end_name, second_point._name.c_str());
//       cts_segment.end_x = second_point._point.x();
//       cts_segment.end_y = second_point._point.y();

//       cts_segment_list.push_back(cts_segment);
//     }
//     _net_map.insert(std::make_pair(net_name, cts_segment_list));
//   }
// }

int32_t CtsIO::get_routing_buffer_size()
{
  int32_t buffer_size = sizeof(CtsFileNetHeader);
  for (auto [net_name, topo_list] : _net_map) {
    buffer_size = buffer_size + sizeof(CtsFileSegmentHeader) + topo_list.size() * sizeof(CtsFileSegment);
  }
  return buffer_size;
}

bool CtsIO::readCtsDataFromFile(string file_path)
{
  if (file_path.empty()) {
    return false;
  }

  FileCtsManager file(file_path, (int32_t) CtsDbId::kCtsRoutingData);

  return file.readFile();
}

bool CtsIO::saveCtsDataToFile(string file_path)
{
  if (file_path.empty()) {
    return false;
  }

  FileCtsManager file(file_path, (int32_t) CtsDbId::kCtsRoutingData);

  return file.writeFile();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int64_t CtsIO::getCtsTreeBufferSize()
{
  int64_t buffer_size = sizeof(CtsFileTreeHeader);
  for (auto& node_map : _node_list) {
    buffer_size = buffer_size + sizeof(CtsFileNodeHeader) + node_map->get_buffer_size();
  }
  return buffer_size;
}

bool CtsIO::readTreeDataFromFile(std::string path)
{
  if (path.empty()) {
    return false;
  }

  FileCtsManager file(path, (int32_t) CtsDbId::kCtsGuiData);

  return file.readFile();
}

bool CtsIO::saveTreeDataToFile(std::string path)
{
  _node_list.clear();

  if (path.empty()) {
    return false;
  }

  getTreeData();

  FileCtsManager file(path, (int32_t) CtsDbId::kCtsGuiData);

  return file.writeFile();
}

vector<CtsTreeNodeMap*>& CtsIO::getTreeData(std::string path)
{
  _node_list.clear();

  if (path.empty()) {
    auto& sta_tree = staInst->getClockTree();
    wrapTree(sta_tree);
    updateLeafNumber();
  } else {
    /// get drc detail data from file
    readTreeDataFromFile(path);
  }

  return _node_list;
}

void CtsIO::wrapTree(std::vector<std::unique_ptr<ista::StaClockTree>>& sta_tree_list)
{
  _node_list.clear();

  for (auto& sta_tree : sta_tree_list) {
    CtsTreeNodeMap* node_map = addTreeNodeMap(sta_tree_list.size());

    /// save root
    auto sta_root_node = sta_tree->get_root_node();
    CtsFileTreeNode file_root_node;
    memset(&file_root_node, 0, sizeof(CtsFileTreeNode));

    CtsTreeNode* root_node = new CtsTreeNode(file_root_node);
    root_node->set_node_name(sta_root_node->get_inst_name_str());

    node_map->addNode(root_node);
    node_map->set_root(root_node);

    /// save child
    for (auto& sta_child_node : sta_tree->get_child_nodes()) {
      /// get delay
      double parent_input_max_rise_AT = 0;
      auto parent_input_max_rise_ATs = sta_child_node->getInputPinMaxRiseAT();
      if (parent_input_max_rise_ATs.size() > 0) {
        parent_input_max_rise_AT = (*parent_input_max_rise_ATs.begin()).second;
      }

      CtsFileTreeNode file_node;
      memset(&file_node, 0, sizeof(CtsFileTreeNode));

      CtsTreeNode* node = new CtsTreeNode(file_node);
      node->set_node_name(sta_child_node->get_inst_name_str());
      node->set_delay(parent_input_max_rise_AT);

      node_map->addNode(node);
    }

    for (auto& sta_child_arc : sta_tree->get_child_arcs()) {
      auto sta_node_parent = sta_child_arc->get_parent_node();
      auto sta_node_child = sta_child_arc->get_child_node();
      if (sta_node_parent == nullptr || sta_node_child == nullptr) {
        std::cout << "[Warning] : No sta node in arc" << std::endl;
        continue;
      }

      auto cts_node_parent = node_map->findNode(sta_node_parent->get_inst_name_str());
      auto cts_node_child = node_map->findNode(sta_node_child->get_inst_name_str());
      if (cts_node_parent == nullptr || cts_node_child == nullptr) {
        continue;
      }

      cts_node_child->set_parent_name(sta_node_parent->get_inst_name_str());

      cts_node_parent->addChildNode(cts_node_child);
    }
  }
}

void CtsIO::updateLeafNumber()
{
  for (auto& node_map : _node_list) {
    auto node = node_map->get_root();

    if (node == nullptr) {
      continue;
    }

    /// if leaf number has been set, ignore
    if (node->get_leaf_num() <= 0) {
      updateTotalNumberForLeaf(node);
    }
  }
}

int64_t CtsIO::updateTotalNumberForLeaf(CtsTreeNode* node)
{
  /// if node is leaf, return 1
  if (node->is_leaf()) {
    return 1;
  }

  /// if leaf has been set, return leaf number
  if (node->get_leaf_num() > 0) {
    return node->get_leaf_num();
  }

  int64_t leaf_num = 0;
  for (auto& child_node : node->get_child_node_list()) {
    leaf_num += updateTotalNumberForLeaf(child_node);
  }

  /// if all leaf find, set node leaf number
  node->set_leaf_num(leaf_num);

  return leaf_num;
}

}  // namespace iplf
