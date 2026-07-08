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

#include <glog/logging.h>

#include "builder.h"
#include "dm_cts_config.h"
#include "flow_config.h"
#include "iCTS/api/CTSAPI.hh"
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

  const auto setup_status = CTS_API_INST.init(config, work_dir);
  if (!setup_status.ok()) {
    LOG(ERROR) << "iCTS setup failed: " << setup_status.message;
    flowConfigInst->add_status_runtime(stats.elapsedRunTime());
    flowConfigInst->set_status_memmory(stats.memoryDelta());
    return false;
  }
  const auto run_status = CTS_API_INST.runCTS();
  if (!run_status.ok()) {
    LOG(ERROR) << "iCTS run failed: " << run_status.message;
    flowConfigInst->add_status_runtime(stats.elapsedRunTime());
    flowConfigInst->set_status_memmory(stats.memoryDelta());
    return false;
  }

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool CtsIO::reportCTS(std::string path)
{
  if (path.empty()) {
    path = flowConfigInst->get_icts_path();
  }
  const auto report_status = CTS_API_INST.report(path);
  if (!report_status.ok()) {
    LOG(ERROR) << "iCTS report failed: " << report_status.message;
    return false;
  }
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
  if (path.empty()) {
    return false;
  }

  FileCtsManager file(path, (int32_t) CtsDbId::kCtsGuiData);

  return file.writeFile();
}

vector<CtsTreeNodeMap*>& CtsIO::getTreeData(std::string path)
{
  _node_list.clear();

  if (path.empty()) {
    return _node_list;
  } else {
    /// get drc detail data from file
    readTreeDataFromFile(path);
  }

  return _node_list;
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
