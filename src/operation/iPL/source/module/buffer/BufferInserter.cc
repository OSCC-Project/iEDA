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
 * @Date: 2022-11-17 09:12:09
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-12-14 17:39:59
 * @FilePath: /irefactor/src/operation/iPL/source/module/buffer/BufferInserter.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置:
 * https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include "BufferInserter.hh"

#include <set>
#include <stack>

#include "module/evaluator/wirelength/HPWirelength.hh"
#include "module/evaluator/wirelength/SteinerWirelength.hh"
#include "module/logger/Log.hh"
#include "usage/usage.hh"

namespace ipl {

void BufferInserter::initBufferConfig(Config* config)
{
  _buffer_config = config->get_buffer_config();
}

void BufferInserter::initMasterList()
{
  auto* ipl_layout = _placer_db->get_layout();
  for (auto master_name : _buffer_config.get_buffer_master_list()) {
    auto* cell_master = ipl_layout->find_cell(master_name);
    if (cell_master) {
      _buffer_master_list.push_back(cell_master);
    }
  }
}

void BufferInserter::runBufferInsertionForMaxWireLength()
{
  LOG_INFO << "-----------------Start Buffer Insertion For Max Wirelength repair-----------------";
  ieda::Stats buffer_status;

  int32_t max_wirelength = _buffer_config.get_max_wirelength_constraint();
  auto* topo_manager = _placer_db->get_topo_manager();
  HPWirelength hpwl_eval(topo_manager);
  SteinerWirelength stwl_eval(topo_manager);

  // obtain violated network list
  std::vector<NetWork*> violated_network_list;
  for (auto* network : topo_manager->get_network_list()) {
    // skip dont care net.
    if (std::fabs(network->get_net_weight()) < 1e-7) {
      continue;
    }
    // skip the clock net.
    if (iPLAPIInst.isClockNet(network->get_name())) {
      continue;
    }
    int32_t hpwl = hpwl_eval.obtainNetWirelength(network->get_network_id());
    if (hpwl > max_wirelength) {
      violated_network_list.push_back(network);
    }
  }

  stwl_eval.updatePartOfNetWorkPointPair(violated_network_list);

  int32_t fixed_net_cnt = 0;
  LOG_INFO << "Violation Net Count: " << violated_network_list.size();
  for (auto* violated_network : violated_network_list) {
    // skip the lack of driven pin net.
    if (!violated_network->get_transmitter()) {
      continue;
    }
    MultiTree* topo_tree = stwl_eval.obtainMultiTree(violated_network);
    if (!insertBufferWithMaxWireLength(topo_tree, 0)) {
      LOG_WARNING << "Fixing Net : " << topo_tree->get_network()->get_name() << " Buffer Insertion Error!";
    }

    fixed_net_cnt++;
    LOG_INFO_IF(fixed_net_cnt % 100 == 0) << "Finish Fixed " << fixed_net_cnt << " Nets";
  }

  LOG_INFO << "Total Insert Buffer Count : " << _buffer_cnt;

  PlacerDBInst.initTopoManager();
  PlacerDBInst.updateTopoManager();

  double time_delta = buffer_status.elapsedRunTime();
  LOG_INFO << "Buffer Insertion Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Buffer Insertion-----------------";
}

bool BufferInserter::insertBufferWithMaxWireLength(MultiTree* topo_tree, int32_t buffer_level)
{
  bool insert_flag = false;
  auto* root_node = topo_tree->get_root();
  std::map<TreeNode*, int32_t> sink_wl_map;
  std::map<TreeNode*, std::vector<TreeNode*>> sink_nodes_map;

  std::stack<TreeNode*> node_stack;
  node_stack.push(root_node);
  _modify_net_list.push_back(topo_tree->get_network()->get_name());
  recursiveRecordBufferInfo(topo_tree->get_network()->get_name(), node_stack, sink_wl_map, sink_nodes_map, buffer_level);

  // Call sta to insert buffer list and check.
  for (auto buffer_record : _buffer_record_list) {
    insert_flag
        = iPLAPIInst.insertSignalBuffer(std::make_pair(buffer_record.net_name, buffer_record.sink_net_name), buffer_record.sink_pins,
                                        std::make_pair(buffer_record.buffer_master_name, buffer_record.buffer_name),
                                        std::make_pair(buffer_record.center_x, buffer_record.center_y));
    if (!insert_flag) {
      break;
    }
  }

  // Update and check placerdb
  if (insert_flag) {
    iPLAPIInst.updatePlacerDB(_buffer_list);
  }

  // Check the placerdb.
  insert_flag = checkBufferInsertion();

  _buffer_list.clear();
  _modify_net_list.clear();
  _buffer_record_list.clear();
  for (auto* node : _buffer_node_list) {
    delete node;
  }
  _buffer_node_list.clear();
  for (auto* tree_node : _buffer_tree_node_list) {
    delete tree_node;
    _buffer_tree_node_list.clear();
  }
  // _pin_to_owner.clear();
  // _pin_to_point.clear();

  return insert_flag;
}

void BufferInserter::recursiveRecordBufferInfo(std::string net_name, std::stack<TreeNode*>& node_stack, int32_t& wirelength,
                                               int32_t buffer_level, std::vector<std::string>& cur_sink_nodes)
{
  while (!node_stack.empty()) {
    auto* cur_tree_node = node_stack.top();
    node_stack.pop();

    if (cur_tree_node->get_child_list().empty()) {
      return;
    }

    for (auto* sink_tree_node : cur_tree_node->get_child_list()) {
      node_stack.push(sink_tree_node);
      recursiveRecordBufferInfo(net_name, node_stack, wirelength, buffer_level, cur_sink_nodes);

      // exist cur_tree_node and sink_tree_node.
      if (sink_tree_node->get_node()) {
        cur_sink_nodes.push_back(sink_tree_node->get_node()->get_name());
      }

      // TODO.
      int32_t horizontal_segment = std::abs(cur_tree_node->get_point().get_x() - sink_tree_node->get_point().get_x());
      int32_t vertical_segment = std::abs(cur_tree_node->get_point().get_y() - sink_tree_node->get_point().get_y());
      int32_t sgement_wirelength = (horizontal_segment + vertical_segment);
      if (sgement_wirelength > _buffer_config.get_max_wirelength_constraint()) {
        // record buffer cnt.
        int32_t buffer_insert_cnt = (sgement_wirelength / _buffer_config.get_max_wirelength_constraint());
        for (int32_t i = 0; i < buffer_insert_cnt; i++) {
          if (_buffer_cnt >= _buffer_config.get_max_buffer_num()) {
            break;
          }
          BufferSinksRecord buffer_record = recordBuffer(net_name, buffer_level, cur_sink_nodes);
          _buffer_record_list.push_back(buffer_record);
          _buffer_list.push_back(buffer_record.buffer_name);
          _modify_net_list.push_back(buffer_record.sink_net_name);
          std::string buffer_input_name = buffer_record.buffer_name + ":" + buffer_record.buffer_input_name;
          // change the sink pin to point.
          _pin_to_point.emplace(buffer_input_name, Point<int32_t>(buffer_record.center_x, buffer_record.center_y));
          _pin_to_owner.emplace(buffer_input_name, buffer_input_name);
          for (std::string pin_name : cur_sink_nodes) {
            auto it = _pin_to_owner.find(pin_name);
            if (it != _pin_to_owner.end()) {
              it->second = buffer_input_name;
            } else {
              LOG_WARNING << "Pin :" << pin_name << " has not been record!";
            }
          }
          cur_sink_nodes.clear();
          cur_sink_nodes.push_back(buffer_input_name);  // TBD: read cell input.
        }
      } else {
        // add wirelength
      }

      // TODO.
      int32_t cur_layer_wl = obtainCurTreeLayerWL(cur_tree_node);
      if (wirelength + cur_layer_wl > _buffer_config.get_max_wirelength_constraint()) {
        // record buffer.
        BufferSinksRecord buffer_record = recordBuffer(net_name, buffer_level, cur_sink_nodes);
      }

      wirelength += sgement_wirelength;
      int32_t buffer_insert_cnt = (wirelength / _buffer_config.get_max_wirelength_constraint());
      for (int32_t i = 0; i < buffer_insert_cnt; i++) {
        if (_buffer_cnt >= _buffer_config.get_max_buffer_num()) {
          break;
        }

        wirelength -= _buffer_config.get_max_wirelength_constraint();
        // // only one direction has value.

        // if(horizontal_segment != 0){
        //   if(cur_tree_node->get_point().get_x() < sink_tree_node->get_point().get_x()){
        //       center_x = cur_tree_node->get_point().get_x() + wirelength;
        //   }else{
        //       center_x = cur_tree_node->get_point().get_x() - wirelength;
        //   }
        //   center_y = cur_tree_node->get_point().get_y();
        // }else{
        //   if(cur_tree_node->get_point().get_y() < sink_tree_node->get_point().get_y()){
        //       center_y = cur_tree_node->get_point().get_y() + wirelength;
        //   }else{
        //       center_y = cur_tree_node->get_point().get_y() - wirelength;
        //   }
        //   center_x = cur_tree_node->get_point().get_x();
        // }
      }
    }
  }
}

void BufferInserter::recursiveRecordBufferInfo(std::string net_name, std::stack<TreeNode*>& node_stack,
                                               std::map<TreeNode*, int32_t>& sink_wl_map,
                                               std::map<TreeNode*, std::vector<TreeNode*>>& sink_nodes_map, int32_t buffer_level)
{
  while (!node_stack.empty()) {
    auto* cur_tree_node = node_stack.top();

    if (cur_tree_node->get_child_list().empty()) {
      // record sink wl.
      sink_wl_map.emplace(cur_tree_node, 0);
      return;
    }

    for (auto* sink_tree_node : cur_tree_node->get_child_list()) {
      node_stack.push(sink_tree_node);
      recursiveRecordBufferInfo(net_name, node_stack, sink_wl_map, sink_nodes_map, buffer_level);

      // record node.
      std::vector<TreeNode*> record_sink_nodes;
      auto record_iter_1 = sink_nodes_map.find(sink_tree_node);
      if (record_iter_1 != sink_nodes_map.end()) {
        record_sink_nodes = record_iter_1->second;
      }
      record_sink_nodes.push_back(sink_tree_node);

      // record wl.
      int32_t cur_wl = obtainPointPairDist(cur_tree_node->get_point(), sink_tree_node->get_point());
      int32_t sink_wl = sink_wl_map.at(sink_tree_node);

      if (sink_wl > _buffer_config.get_max_wirelength_constraint()) {
        LOG_ERROR << "ERROR IN SINK WL";
      }

      int32_t record_wl = cur_wl + sink_wl;

      // maybe double or trible the max_wirelength.
      int32_t buffer_insert_cnt = (record_wl / _buffer_config.get_max_wirelength_constraint());
      TreeNode* cur_sink_tree_node = sink_tree_node;
      for (int32_t i = 0; i < buffer_insert_cnt; i++) {
        // record buffer.
        int32_t wl_delta = _buffer_config.get_max_wirelength_constraint() - sink_wl;
        // find real node among record_sink_nodes.
        std::vector<std::string> wrap_nodes = findTrueNodeAmongTreeNodes(record_sink_nodes);
        BufferSinksRecord buffer_record = recordBuffer(
            net_name, std::make_pair(cur_tree_node->get_point(), cur_sink_tree_node->get_point()), wl_delta, buffer_level, wrap_nodes);
        sink_wl = 0;
        record_wl -= wl_delta;

        // TODO: simplify the record.
        Node* buffer_node = new Node(buffer_record.buffer_name + ":" + buffer_record.buffer_input_name);
        TreeNode* buffer_tree_node = new TreeNode(Point<int32_t>(buffer_record.center_x, buffer_record.center_y));
        buffer_tree_node->set_node(buffer_node);
        _buffer_node_list.push_back(buffer_node);
        _buffer_tree_node_list.push_back(buffer_tree_node);

        cur_sink_tree_node = buffer_tree_node;
        record_sink_nodes.clear();
        record_sink_nodes.push_back(buffer_tree_node);
        _buffer_record_list.push_back(buffer_record);
        _buffer_list.push_back(buffer_record.buffer_name);
        _modify_net_list.push_back(buffer_record.sink_net_name);
      }

      // deal with the sum case.
      int32_t sink_tree_node_cnt = cur_tree_node->get_child_list().size();
      int32_t max_wl_for_source = _buffer_config.get_max_wirelength_constraint() / sink_tree_node_cnt;
      if (record_wl > max_wl_for_source) {
        int32_t wl_delta = record_wl - max_wl_for_source;
        // find real node among record_sink_nodes.
        std::vector<std::string> wrap_nodes = findTrueNodeAmongTreeNodes(record_sink_nodes);
        BufferSinksRecord buffer_record = recordBuffer(
            net_name, std::make_pair(cur_tree_node->get_point(), cur_sink_tree_node->get_point()), wl_delta, buffer_level, wrap_nodes);
        record_wl -= wl_delta;

        // TODO: simplify the record.
        Node* buffer_node = new Node(buffer_record.buffer_name + ":" + buffer_record.buffer_input_name);
        TreeNode* buffer_tree_node = new TreeNode(Point<int32_t>(buffer_record.center_x, buffer_record.center_y));
        buffer_tree_node->set_node(buffer_node);
        _buffer_node_list.push_back(buffer_node);
        _buffer_tree_node_list.push_back(buffer_tree_node);

        // NOLINTNEXTLINE
        cur_sink_tree_node = buffer_tree_node;
        record_sink_nodes.clear();
        record_sink_nodes.push_back(buffer_tree_node);
        _buffer_record_list.push_back(buffer_record);
        _buffer_list.push_back(buffer_record.buffer_name);
        _modify_net_list.push_back(buffer_record.sink_net_name);
      }

      auto record_iter_2 = sink_wl_map.find(cur_tree_node);
      if (record_iter_2 != sink_wl_map.end()) {
        record_iter_2->second += record_wl;
      } else {
        sink_wl_map.emplace(cur_tree_node, record_wl);
      }

      auto record_iter_3 = sink_nodes_map.find(cur_tree_node);
      if (record_iter_3 != sink_nodes_map.end()) {
        (record_iter_3->second).insert(record_iter_3->second.end(), record_sink_nodes.begin(), record_sink_nodes.end());
      } else {
        sink_nodes_map.emplace(cur_tree_node, record_sink_nodes);
      }
    }
  }

  node_stack.pop();
}

std::vector<std::string> BufferInserter::findTrueNodeAmongTreeNodes(std::vector<TreeNode*>& tree_node_list)
{
  std::vector<std::string> true_nodes;
  for (auto* tree_node : tree_node_list) {
    if (tree_node->get_node()) {
      true_nodes.push_back(tree_node->get_node()->get_name());
    }
  }
  return true_nodes;
}
BufferSinksRecord BufferInserter::recordBuffer(std::string net_name, int32_t buffer_level, std::vector<std::string>& sink_nodes)
{
  // add buffer.
  BufferSinksRecord buffer_record(_buffer_cnt++);
  auto* buffer_master = _buffer_master_list.at(buffer_level);
  buffer_record.buffer_master_name = buffer_master->get_name();
  buffer_record.net_name = net_name;
  buffer_record.sink_pins.assign(sink_nodes.begin(), sink_nodes.end());

  // calculate the optimal coordinate.
  std::vector<Point<int32_t>> point_list;
  std::vector<Point<int32_t>> sink_point_list;
  for (auto pair : _pin_to_owner) {
    if (pair.first != pair.second) {
      continue;
    }
    point_list.push_back(_pin_to_point.at(pair.first));
  }
  Point<int32_t> optimal_point = obtainOptimalPoint(point_list);
  for (auto sink_node : sink_nodes) {
    sink_point_list.push_back(_pin_to_point.at(sink_node));
  }
  Point<int32_t> boundbox_point = moveToBoundingBox(optimal_point, sink_point_list);

  int32_t center_x = boundbox_point.get_x();
  int32_t center_y = boundbox_point.get_y();

  // check if inside the core.
  auto core_shape = _placer_db->get_layout()->get_core_shape();
  center_x < core_shape.get_ll_x() ? center_x = core_shape.get_ll_x() + buffer_master->get_width() : center_x;
  center_y < core_shape.get_ll_y() ? center_y = core_shape.get_ll_y() + buffer_master->get_height() : center_y;
  center_x > core_shape.get_ur_x() ? center_x = core_shape.get_ur_x() - buffer_master->get_width() : center_x;
  center_y > core_shape.get_ur_y() ? center_y = core_shape.get_ur_y() - buffer_master->get_height() : center_y;
  buffer_record.center_x = center_x;
  buffer_record.center_y = center_y;

  return buffer_record;
}

BufferSinksRecord BufferInserter::recordBuffer(std::string net_name, std::pair<Point<int32_t>, Point<int32_t>> source_sink_pair,
                                               int32_t delta, int32_t buffer_level, std::vector<std::string>& sink_nodes)
{
  // make sure the point is one dimension.
  BufferSinksRecord buffer_record(_buffer_cnt++);

  auto* buffer_master = _buffer_master_list.at(buffer_level);

  // get buffer_inpin_name
  auto inpin_name_list = buffer_master->get_inpin_name_list();
  if (static_cast<int32_t>(inpin_name_list.size()) != 1) {
    LOG_WARNING << "Buffer " << buffer_master->get_name() << " has not only one input pin";
  }
  buffer_record.buffer_input_name = inpin_name_list.at(0);

  buffer_record.buffer_master_name = buffer_master->get_name();
  buffer_record.net_name = net_name;
  buffer_record.sink_pins.assign(sink_nodes.begin(), sink_nodes.end());

  // calculate buffer loc.
  int32_t center_x = source_sink_pair.first.get_x();
  int32_t center_y = source_sink_pair.first.get_y();
  int32_t dx = source_sink_pair.first.get_x() - source_sink_pair.second.get_x();
  if (dx != 0) {
    dx < 0 ? center_x = source_sink_pair.second.get_x() - delta : center_x = source_sink_pair.second.get_x() + delta;
  }
  int32_t dy = source_sink_pair.first.get_y() - source_sink_pair.second.get_y();
  if (dy != 0) {
    dy < 0 ? center_y = source_sink_pair.second.get_y() - delta : center_y = source_sink_pair.second.get_y() + delta;
  }

  // check if inside the core.
  auto core_shape = _placer_db->get_layout()->get_core_shape();
  center_x < core_shape.get_ll_x() ? center_x = core_shape.get_ll_x() + buffer_master->get_width() : center_x;
  center_y < core_shape.get_ll_y() ? center_y = core_shape.get_ll_y() + buffer_master->get_height() : center_y;
  center_x + buffer_master->get_width() > core_shape.get_ur_x() ? center_x = core_shape.get_ur_x() - buffer_master->get_width() : center_x;
  center_y + buffer_master->get_height() > core_shape.get_ur_y() ? center_y = core_shape.get_ur_y() - buffer_master->get_height()
                                                                 : center_y;
  buffer_record.center_x = center_x;
  buffer_record.center_y = center_y;

  return buffer_record;
}

int32_t BufferInserter::obtainCurTreeLayerWL(TreeNode* source)
{
  int32_t wirelength = 0;
  for (auto* child : source->get_child_list()) {
    int32_t horizontal_segment = std::abs(source->get_point().get_x() - child->get_point().get_x());
    int32_t vertical_segment = std::abs(source->get_point().get_y() - child->get_point().get_y());
    wirelength += (horizontal_segment + vertical_segment);
  }
  return wirelength;
}

int32_t BufferInserter::obtainPointPairDist(Point<int32_t> point_1, Point<int32_t> point_2)
{
  int32_t horizontal_segment = std::abs(point_1.get_x() - point_2.get_x());
  int32_t vertical_segment = std::abs(point_1.get_y() - point_2.get_y());
  return (horizontal_segment + vertical_segment);
}

Point<int32_t> BufferInserter::obtainOptimalPoint(std::vector<Point<int32_t>>& point_list)
{
  float optimal_x = 0.0f;
  float optimal_y = 0.0f;
  int32_t center_idx = point_list.size() / 2;

  std::vector<int32_t> x_list;
  std::vector<int32_t> y_list;
  for (auto point : point_list) {
    x_list.push_back(point.get_x());
    y_list.push_back(point.get_y());
  }
  std::sort(x_list.begin(), x_list.end());
  std::sort(y_list.begin(), y_list.end());

  optimal_x = static_cast<float>(x_list.at(center_idx - 1) + x_list.at(center_idx)) / 2;
  optimal_y = static_cast<float>(y_list.at(center_idx - 1) + y_list.at(center_idx)) / 2;

  int32_t coord_x = optimal_x;
  int32_t coord_y = optimal_y;

  return Point<int32_t>(coord_x, coord_y);
}

Point<int32_t> BufferInserter::moveToBoundingBox(Point<int32_t> origin_loc, std::vector<Point<int32_t>>& point_list)
{
  int32_t llx = INT32_MAX;
  int32_t lly = INT32_MAX;
  int32_t urx = INT32_MIN;
  int32_t ury = INT32_MIN;

  for (auto point : point_list) {
    point.get_x() < llx ? llx = point.get_x() : llx;
    point.get_x() > urx ? urx = point.get_x() : urx;
    point.get_y() < lly ? lly = point.get_y() : lly;
    point.get_y() > ury ? ury = point.get_y() : ury;
  }

  if (urx < llx || ury < lly) {
    LOG_WARNING << "Warnning point list!";
  }

  int32_t max_wl = _buffer_config.get_max_wirelength_constraint();
  int32_t new_x = origin_loc.get_x();
  int32_t new_y = origin_loc.get_y();

  if ((urx - llx + ury - lly) > max_wl) {
    // LOG_WARNING << "Sink points Bounding Box has exceed max wirelength constraint!";
    origin_loc.get_x() < llx ? new_x = llx : llx;
    origin_loc.get_x() > urx ? new_x = urx : urx;
    origin_loc.get_y() < lly ? new_y = lly : lly;
    origin_loc.get_y() > ury ? new_y = ury : ury;
  } else {
    // case 1.
    if (origin_loc.get_x() < llx) {
      int32_t cota_x = max_wl - llx + origin_loc.get_x();
      if (cota_x > 0) {
        if (origin_loc.get_y() < lly) {
          int32_t cota_y = cota_x - lly + origin_loc.get_y();
          if (cota_y < 0) {
            new_y = origin_loc.get_y() - cota_y;
          }
        }
        if (origin_loc.get_y() > ury) {
          int32_t cota_y = cota_x + ury - origin_loc.get_y();
          if (cota_y < 0) {
            new_y = origin_loc.get_y() + cota_y;
          }
        }
      } else {
        new_x = origin_loc.get_x() - cota_x;
        if (origin_loc.get_y() < lly) {
          new_y = lly;
        }
        if (origin_loc.get_y() > ury) {
          new_y = ury;
        }
      }
    }

    // case 2.
    if (origin_loc.get_x() > urx) {
      int32_t cota_x = max_wl + urx - origin_loc.get_x();
      if (cota_x > 0) {
        if (origin_loc.get_y() < lly) {
          int32_t cota_y = cota_x - lly + origin_loc.get_y();
          if (cota_y < 0) {
            new_y = origin_loc.get_y() - cota_y;
          }
        }
        if (origin_loc.get_y() > ury) {
          int32_t cota_y = cota_x + ury - origin_loc.get_y();
          if (cota_y < 0) {
            new_y = origin_loc.get_y() + cota_y;
          }
        }
      } else {
        new_x = origin_loc.get_x() + cota_x;
        if (origin_loc.get_y() < lly) {
          new_y = lly;
        }
        if (origin_loc.get_y() > ury) {
          new_y = ury;
        }
      }
    }

    // case 3.
    if (origin_loc.get_x() >= llx && origin_loc.get_x() <= urx) {
      if (origin_loc.get_y() < lly) {
        int32_t cota_y = max_wl - lly + origin_loc.get_y();
        if (cota_y < 0) {
          new_y = origin_loc.get_y() - cota_y;
        }
      }
      if (origin_loc.get_y() > ury) {
        int32_t cota_y = max_wl + ury - origin_loc.get_y();
        if (cota_y < 0) {
          new_y = origin_loc.get_y() + cota_y;
        }
      }
    }
  }

  return Point<int32_t>(new_x, new_y);
}

bool BufferInserter::checkBufferInsertion()
{
  auto* ipl_design = _placer_db->get_design();

  for (std::string buffer_name : _buffer_list) {
    auto* buffer = ipl_design->find_instance(buffer_name);
    if (!buffer) {
      LOG_WARNING << buffer_name << " has not been correct inserted!";
      return false;
    }
  }

  for (std::string net_name : _modify_net_list) {
    auto* net = ipl_design->find_net(net_name);
    if (!net) {
      LOG_WARNING << net_name << " has not been correct add!";
      return false;
    }
  }
  return true;
}

}  // namespace ipl