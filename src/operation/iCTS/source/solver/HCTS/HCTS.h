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
 * @file HCts.h
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#pragma once
#include <string>
#include <vector>

#include "CTSAPI.hpp"
#include "ClockTopo.h"
#include "CtsCellLib.h"
#include "CtsConfig.h"
#include "CtsInstance.h"
#include "log/Log.hh"

namespace icts {
enum class HNodeType
{
  kSink,
  kBuffer,
  kSteiner
};

class HNode
{
 public:
  HNode() { _id = CTSAPIInst.genId(); }

  HNode(CtsInstance* inst) : _inst(inst)
  {
    _type = inst->get_type() == CtsInstanceType::kBuffer ? HNodeType::kBuffer : HNodeType::kSink;
    _id = CTSAPIInst.genId();
  }

  HNode(HNode* left, HNode* right)
  {
    _left = left;
    _right = right;
    _left->set_parent(this);
    _right->set_parent(this);
    _id = CTSAPIInst.genId();
    _inst = new CtsInstance("steiner_" + std::to_string(_id), "", CtsInstanceType::kSteinerPoint, Point(-1, -1));
  }

  ~HNode() = default;

  // get
  std::string getName() const { return _inst->get_name(); }
  Point getLocation() const { return _inst->get_location(); }
  std::string getCellMaster() const { return _inst->get_cell_master(); }

  CtsInstance* get_inst() const { return _inst; }
  double get_delay() const { return _delay; }
  double get_insertion_delay() const { return _insertion_delay; }
  double get_slew_in() const { return _slew_in; }
  double get_cap_load() const { return _cap_load; }
  double get_cap_out() const { return _cap_out; }
  double get_feasible_cap() const { return _feasible_cap; }
  double get_sub_total_cap() const { return _sub_total_cap; }
  int get_fanout() const { return _fanout; }
  int get_level() const { return _level; }
  HNode* get_parent() const { return _parent; }
  HNode* get_left() const { return _left; }
  HNode* get_right() const { return _right; }
  HNodeType get_type() const { return _type; }
  int get_id() const { return _id; }
  double get_net_length() const { return _net_length; }

  // set
  void setName(const std::string& name) { _inst->set_name(name); }
  void setLocation(const Point& location) { _inst->set_location(location); }
  void setCellMaster(const std::string& cell_master) { _inst->set_cell_master(cell_master); }

  void set_inst(CtsInstance* inst) { _inst = inst; }
  void set_delay(const double& delay) { _delay = delay; }
  void set_insertion_delay(const double& insertion_delay) { _insertion_delay = insertion_delay; }
  void set_slew_in(const double& slew_in) { _slew_in = slew_in; }
  void set_cap_load(const double& cap_load) { _cap_load = cap_load; }
  void set_cap_out(const double& cap_out) { _cap_out = cap_out; }
  void set_feasible_cap(const double& feasible_cap) { _feasible_cap = feasible_cap; }
  void set_sub_total_cap(const double& sub_total_cap) { _sub_total_cap = sub_total_cap; }
  void set_fanout(const size_t& fanout) { _fanout = fanout; }
  void set_level(const size_t& level) { _level = level; }
  void set_parent(HNode* parent) { _parent = parent; }
  void set_left(HNode* left) { _left = left; }
  void set_right(HNode* right) { _right = right; }
  void set_type(const HNodeType& type)
  {
    _type = type;
    switch (type) {
      case HNodeType::kSink:
        _inst->set_type(CtsInstanceType::kSink);
        break;
      case HNodeType::kBuffer:
        _inst->set_type(CtsInstanceType::kBuffer);
        break;
      case HNodeType::kSteiner:
        _inst->set_type(CtsInstanceType::kSteinerPoint);
        break;
      default:
        break;
    }
  }
  void set_id(const size_t& id) { _id = id; }
  void set_net_length(const double& net_length) { _net_length = net_length; }

  // bool
  bool isSink() const { return _type == HNodeType::kSink; }
  bool isBuffer() const { return _type == HNodeType::kBuffer; }
  bool isSteiner() const { return _type == HNodeType::kSteiner; }

 private:
  CtsInstance* _inst = nullptr;
  double _delay = 0;
  double _insertion_delay = 0;
  double _slew_in = 0;
  double _cap_load = 0;
  double _cap_out = 0;
  double _feasible_cap = 0;
  double _sub_total_cap = 0;
  size_t _fanout = 1;
  size_t _level = 1;
  HNode* _parent = nullptr;
  HNode* _left = nullptr;
  HNode* _right = nullptr;
  HNodeType _type = HNodeType::kSteiner;
  size_t _id = 0;
  double _net_length = 0.0;
};

class HCTS
{
 public:
  HCTS() = delete;
  HCTS(const std::string& net_name, const std::vector<CtsInstance*>& instances) : _net_name(net_name), _instances(instances)
  {
    auto* config = CTSAPIInst.get_config();
    // unit
    _unit_res = CTSAPIInst.getClockUnitRes() / 1000;
    _unit_cap = CTSAPIInst.getClockUnitCap();
    _db_unit = CTSAPIInst.getDbUnit();
    // constraint
    _skew_bound = config->get_skew_bound();
    _max_cap = config->get_max_cap();
    _max_buf_tran = config->get_max_buf_tran();
    _max_sink_tran = config->get_max_sink_tran();
    _max_fanout = config->get_max_fanout();
    _max_length = config->get_max_length();
    // lib
    _delay_libs = CTSAPIInst.getAllBufferLibs();
    // temp default buf
    _lib = _delay_libs.front();

    // run
    run();
  }

  ~HCTS() = default;

  // get
  std::vector<ClockTopo> get_clk_topos() const { return _clock_topos; }

  // find
  HNode* findNode(const std::string& node_name) const
  {
    auto it = _node_map.find(node_name);
    if (it != _node_map.end()) {
      return it->second;
    }
    LOG_FATAL << "Can't find node: " << node_name;
    return nullptr;
  }

 private:
  void run();
  // func
  HNode* mergeNode(HNode* left, HNode* right) const;

  HNode* biCluster(const std::vector<CtsInstance*>& insts) const;

  std::vector<std::vector<CtsInstance*>> kMeans(const std::vector<CtsInstance*>& instances, const size_t& k,
                                                const size_t& max_iter = 10) const;

  HNode* biPartition(const std::vector<CtsInstance*>& instances) const;

  Point medianCenter(const std::vector<CtsInstance*>& instances) const;

  Point meanCenter(const std::vector<CtsInstance*>& instances) const;

  void netPropagation(HNode* node) const;

  void timingPropagation(HNode* root) const;

  void netLengthPropagation(HNode* node) const;

  void fanoutPropagation(HNode* node) const;

  void capPropagation(HNode* node) const;

  void slewPropagation(HNode* node) const;

  void heuristicBuffering() const;

  void recursiveBuffering(HNode* node) const;

  void allocateRemainCap(HNode* node) const;

  HNode* capFeasibleNode(HNode* parent, HNode* child) const;

  void makeTopos();

  ClockTopo makeTopo(HNode* root) const;

  // instantiation
  HNode* genBufferNode() const;

  void setBuffer(HNode* node) const;

  HNode* makeBuffer(HNode* parent, HNode* child, const Point& loc) const;

  void connect(HNode* top, HNode* mid, HNode* bottom) const;

  // basic update
  void updateSubTotalCap(HNode* node) const;
  void updateCapCenterLoc(HNode* node) const;

  // basic calc
  int calcManhattanDist(const Point& p1, const Point& p2) const { return std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y()); }
  double calcLength(HNode* node1, HNode* node2) const { return calcLength(node1->getLocation(), node2->getLocation()); }
  double calcLength(const Point& p1, const Point& p2) const { return 1.0 * calcManhattanDist(p1, p2) / _db_unit; }
  Point internalPoint(const Point& p1, const Point& p2, const int& dist_to_left) const
  {
    LOG_FATAL_IF(dist_to_left < 0) << "dist to left should be positive, but got " << dist_to_left;
    auto dist = calcManhattanDist(p1, p2);
    auto ratio = 1.0 * dist_to_left / dist;
    auto loc = p1 + (p2 - p1) * ratio;
    return loc;
  }
  double calcElmoreDelay(HNode* parent, HNode* child) const
  {
    auto length = calcLength(parent, child);
    auto delay = _unit_res * length * (_unit_cap * length / 2 + child->get_cap_out());
    return delay;
  }
  double calcIdealSlew(HNode* parent, HNode* child) const { return std::log(9) * calcElmoreDelay(parent, child); }
  double calcGeometricMean(const double& p, const double& q) const { return std::sqrt(std::pow(p, 2) + std::pow(q, 2)); }

  // report

  void reportTiming() const;
  // member
  std::string _net_name;
  std::vector<CtsInstance*> _instances;
  HNode* _root = nullptr;
  std::vector<ClockTopo> _clock_topos;
  std::map<std::string, HNode*> _node_map;

  // design info
  // unit
  double _unit_res = 0.0;
  double _unit_cap = 0.0;
  size_t _db_unit = 0;
  // constraint
  double _skew_bound = 0.0;
  double _max_cap = 0.0;
  double _max_buf_tran = 0.0;
  double _max_sink_tran = 0.0;
  size_t _max_fanout = 0;
  double _max_length = 0;
  // lib
  std::vector<CtsCellLib*> _delay_libs;
  // temp default buf
  CtsCellLib* _lib = nullptr;
};

}  // namespace icts