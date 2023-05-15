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
 * @Date: 2022-11-17 09:03:08
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-12-14 11:10:09
 * @FilePath: /irefactor/src/operation/iPL/source/module/buffer/BufferInserter.hh
 */

#ifndef IPL_BUFFER_INSERTER_H
#define IPL_BUFFER_INSERTER_H

#include <stack>
#include <string>
#include <vector>

#include "PlacerDB.hh"
#include "config/BufferInserterConfig.hh"
#include "PLAPI.hh"

namespace ipl {

struct BufferSinksRecord
{
  BufferSinksRecord(int32_t id)
      : buffer_id(id),
        net_name(""),
        sink_net_name("PLACE_net_" + std::to_string(buffer_id)),
        buffer_name("buffer_" + std::to_string(buffer_id)),
        buffer_master_name(""),
        buffer_input_name(""),
        center_x(-1),
        center_y(-1)
  {
  }

  int32_t buffer_id;
  std::string net_name;
  std::string sink_net_name;
  std::vector<std::string> sink_pins;
  std::string buffer_name;
  std::string buffer_master_name;
  std::string buffer_input_name;
  int32_t center_x;
  int32_t center_y;
};

class MultiTree;

class BufferInserter
{
 public:
  BufferInserter() = delete;
  BufferInserter(Config* config, PlacerDB* placer_db);
  BufferInserter(const BufferInserter&) = delete;
  BufferInserter(BufferInserter&&) = delete;
  ~BufferInserter() = default;

  BufferInserter& operator=(const BufferInserter&) = delete;
  BufferInserter& operator=(BufferInserter&&) = delete;

  void add_buffer_master_list(Cell* master) { _buffer_master_list.push_back(master); }

  void runBufferInsertionForMaxWireLength();

 private:
  int32_t _buffer_cnt;
  BufferInserterConfig _buffer_config;
  PlacerDB* _placer_db;
  std::vector<std::string> _buffer_list;
  std::vector<Cell*> _buffer_master_list;

  // tmp for every violated nets
  std::vector<std::string> _modify_net_list;
  std::vector<BufferSinksRecord> _buffer_record_list;
  std::vector<Node*> _buffer_node_list;
  std::vector<TreeNode*> _buffer_tree_node_list;
  std::map<std::string, Point<int32_t>> _pin_to_point;
  std::map<std::string, std::string> _pin_to_owner;

  void initBufferConfig(Config* config);
  void initMasterList();
  bool insertBufferWithMaxWireLength(MultiTree* topo_tree, int32_t buffer_level);
  void recursiveRecordBufferInfo(std::string net_name, std::stack<TreeNode*>& node_stack, int32_t& cur_wirelength, int32_t buffer_level,
                                 std::vector<std::string>& cur_sink_nodes);
  void recursiveRecordBufferInfo(std::string net_name, std::stack<TreeNode*>& node_stack, std::map<TreeNode*, int32_t>& sink_wl,
                                 std::map<TreeNode*, std::vector<TreeNode*>>& sink_nodes, int32_t buffer_level);
  Point<int32_t> obtainOptimalPoint(std::vector<Point<int32_t>>& point_list);
  bool checkBufferInsertion();
  Point<int32_t> moveToBoundingBox(Point<int32_t> orgin_loc, std::vector<Point<int32_t>>& point_list);
  int32_t obtainCurTreeLayerWL(TreeNode* source);
  int32_t obtainPointPairDist(Point<int32_t> point_1, Point<int32_t> point_2);
  BufferSinksRecord recordBuffer(std::string net_name, int32_t buffer_level, std::vector<std::string>& sink_nodes);
  BufferSinksRecord recordBuffer(std::string net_name, std::pair<Point<int32_t>, Point<int32_t>> source_sink_pair, int32_t delta,
                                 int32_t buffer_level, std::vector<std::string>& sink_nodes);
  std::vector<std::string> findTrueNodeAmongTreeNodes(std::vector<TreeNode*>& tree_node_list);
};
inline BufferInserter::BufferInserter(Config* config, PlacerDB* placer_db) : _buffer_cnt(0)
{
  _placer_db = placer_db;
  initBufferConfig(config);
  initMasterList();
}

}  // namespace ipl

#endif