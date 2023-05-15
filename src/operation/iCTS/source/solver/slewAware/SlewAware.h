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
 * @file SlewAware.h
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <span>
#include <vector>

#include "ClockTopo.h"
#include "CostCalculator.h"
#include "CtsConfig.h"
#include "CtsPolygon.h"
#include "TimingCalculator.h"

namespace icts {
class SlewAware
{
 public:
  SlewAware() = delete;
  SlewAware(const std::string& net_name, const std::vector<CtsInstance*>& instances) : _net_name(net_name), _instances(instances) {}

  ~SlewAware() = default;

  TimingNode* slewAwareBase(std::vector<TimingNode*>& timing_nodes, TimingCalculator* timer);

  void slewAware();

  void slewAwareByKmeans();

  void slewAwareByBiCluster();

  std::vector<ClockTopo> get_clk_topos() const { return _clock_topos; }

  TimingNode* findTimingNode(const std::string& buffer_name);

  TimingCalculator* genTimer() const;

  std::string getInsertTypeStr(TimingNode* node) const;

  std::string getInsertTypeEncodeStr(TimingNode* node) const;

  void timingLog() const;

  std::pair<double, double> calcNetLength(TimingNode* node) const;
  
  void saveTrainingData() const;

  void recursiveMerge(TimingNode* root, TimingCalculator* timer);

  void costBuild(const std::vector<TimingNode*>& timingNodes, CostCalculator* cost_cal, TimingCalculator* timer);

  void topDown(TimingNode* root);

  void buildClockTopo(TimingNode* root, const std::string& net_name);

  ClockTopo makeTopo(TimingNode* root, const std::string& net_name);

  std::vector<TimingNode*> findDrivers(TimingNode* root);

  std::vector<std::vector<CtsInstance*>> clustering(const std::vector<CtsInstance*>& insts, const int& cluster_size = 50);

  TimingNode* biCluster(const std::vector<CtsInstance*>& insts);

  vector<vector<CtsInstance*>> manhattanKmeans(const vector<CtsInstance*>& instances, const int& k, const int& max_iterations = 100);

  std::string _net_name;
  std::vector<CtsInstance*> _instances;
  std::vector<ClockTopo> _clock_topos;
  std::map<std::string, TimingNode*> _timing_node_map;
};
template <>
struct DataTraits<CtsInstance*>
{
  typedef CtsPoint<Coordinate> point_type;
  typedef std::string id_type;

  static inline id_type getId(CtsInstance* inst) { return inst->get_name(); }
  static inline Coordinate getX(CtsInstance* inst) { return inst->get_location().x(); }
  static inline Coordinate getY(CtsInstance* inst) { return inst->get_location().y(); }
  static inline point_type getPoint(CtsInstance* inst) { return inst->get_location(); }
};
}  // namespace icts