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

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "ClockTopo.h"
#include "CtsConfig.h"
#include "CtsDesign.h"
#include "CtsInstance.h"
#include "CtsNet.h"
#include "CtsSignalWire.h"
#include "DME.h"
#include "EvalNet.h"
#include "Kmeans.h"
#include "Params.h"
#include "SkewScheduler.h"
#include "SlewAware.h"
#include "Topologize.h"
#include "Topology.h"
#include "Traits.h"
#include "pgl.h"
namespace icts {

template <>
struct TimeTraits<ZstNode<Endpoint>>
{
  static inline double getTime(const ZstNode<Endpoint>& t)
  {
    auto delay = t.get_delay();
    return delay._time;
  }
  static inline void setTime(ZstNode<Endpoint>& t, double time)
  {
    auto delay = t.get_delay();
    delay._time = time;
    t.set_delay(delay);
  }
};
// template <>
// struct DataTraits<CtsInstance*> {
//   typedef CtsPoint<Coordinate> point_type;
//   typedef std::string id_type;

//   static inline id_type getId(CtsInstance* inst) { return inst->get_name();
//   } static inline Coordinate getX(CtsInstance* inst) {
//     return inst->get_location().x();
//   }
//   static inline Coordinate getY(CtsInstance* inst) {
//     return inst->get_location().y();
//   }
//   static inline point_type getPoint(CtsInstance* inst) {
//     return inst->get_location();
//   }
// };

template <>
struct DataTraits<ZstNode<Endpoint>>
{
  typedef CtsPoint<Coordinate> point_type;
  typedef std::string id_type;

  static inline id_type getId(ZstNode<Endpoint>& node) { return node.get_data()._name; }
  static inline Coordinate getX(ZstNode<Endpoint>& node) { return node.get_data()._point.x(); }
  static inline Coordinate getY(ZstNode<Endpoint>& node) { return node.get_data()._point.y(); }
  static inline point_type getPoint(ZstNode<Endpoint>& node) { return node.get_data()._point; }

  static inline void setId(ZstNode<Endpoint>& node, const std::string& name) { node.get_data()._name = name; }
  static inline void setX(ZstNode<Endpoint>& node, Coordinate coord) { node.get_data()._point.x(coord); }
  static inline void setY(ZstNode<Endpoint>& node, Coordinate coord) { node.get_data()._point.y(coord); }
  static inline void setPoint(ZstNode<Endpoint>& node, const point_type& point) { node.get_data()._point = point; }
};

template <>
struct DataTraits<BstNode<Endpoint>>
{
  typedef CtsPoint<Coordinate> point_type;
  typedef std::string id_type;

  static inline id_type getId(BstNode<Endpoint>& node) { return node.get_data()._name; }
  static inline Coordinate getX(BstNode<Endpoint>& node) { return node.get_data()._point.x(); }
  static inline Coordinate getY(BstNode<Endpoint>& node) { return node.get_data()._point.y(); }
  static inline point_type getPoint(BstNode<Endpoint>& node) { return node.get_data()._point; }

  static inline void setId(BstNode<Endpoint>& node, const std::string& name) { node.get_data()._name = name; }
  static inline void setX(BstNode<Endpoint>& node, Coordinate coord) { node.get_data()._point.x(coord); }
  static inline void setY(BstNode<Endpoint>& node, Coordinate coord) { node.get_data()._point.y(coord); }
  static inline void setPoint(BstNode<Endpoint>& node, const point_type& point) { node.get_data()._point = point; }
};

class Router
{
 public:
  using SkewConstraintMap = std::map<std::pair<std::string, std::string>, std::pair<double, double>>;
  Router() = default;
  Router(const Router&) = default;
  ~Router() = default;
  void init();
  void update();
  void build();
  void DMEBuild();
  void slewAwareBuild();
  void hctsBuild();
  void gocaBuild();
  template <typename T>
  void topoligize(Topology<T>& topo, const std::vector<CtsInstance*>& cluster) const
  {
    LOG_FATAL_IF(cluster.empty()) << "cluster is empty";
    std::vector<T> datas;
    for (auto* inst : cluster) {
      T data;
      DataTraits<T>::setPoint(data, inst->get_location());
      DataTraits<T>::setId(data, inst->get_name());
      DataTraits<T>::setSubWirelength(data, inst->getSubWirelength());
      datas.emplace_back(data);
    }

    if (cluster.size() == 1) {
      T val;
      DataTraits<T>::setPoint(val, cluster.front()->get_location());
      DataTraits<T>::setSubWirelength(val, cluster.front()->getSubWirelength());
      std::vector<TopoNode<T>> nodes = {TopoNode<T>{datas.front(), 1, -1, -1, cluster.front()->getSubWirelength()},
                                        TopoNode<T>{val, -1, 0, -1, cluster.front()->getSubWirelength()}};
      topo = Topology<T>(nodes, 1);
    } else {
      icts::build_topo(topo, datas);
    }
  }

  template <typename T>
  void dme(Topology<T>& topo) const;

  template <typename T>
  void rerouteDME(Topology<T>& topo) const
  {
    auto* config = CTSAPIInst.get_config();
    std::string delay_type = config->get_delay_type();
    DelayModel delay_model;
    if (delay_type == "elmore") {
      delay_model = DelayModel::kElmore;
    } else {
      delay_model = DelayModel::kLinear;
    }
    ZstParams params(delay_model, CTSAPIInst.getDbUnit(), CTSAPIInst.getClockUnitRes(), CTSAPIInst.getClockUnitCap());
    icts::dme(topo, params);
  }

 private:
  void printLog();
  bool haveSink(CtsNet* net) const;

  void routing(CtsNet* clock_net);
  void comfortRouting(CtsNet* clock_net);
  void slewAwareRouting(CtsNet* clock_net);
  void hctsRouting(CtsNet* clk_net);
  void gocaRouting(CtsNet* clk_net);
  void clustering(std::vector<std::vector<CtsInstance*>>& clusters, const std::vector<CtsInstance*>& insts) const;
  int calFeasibleFanout(const double& avg_wirelength) const;
  template <typename T>
  double calAvgWirelength(const int& root_id, Topology<T>& topo) const;
  template <typename T>
  Topology<T> cutTopo(const int& root_id, Topology<T>& topo) const;
  template <typename T>
  std::vector<Topology<T>> splitTopo(Topology<T>& topo, const std::string& net_name) const;
  template <typename T = Endpoint>
  Topology<T> biClusterTopo(const std::vector<CtsInstance*>& insts) const;
  template <typename T>
  TopoNode<T> biCluster(const std::vector<CtsInstance*>& insts, std::vector<TopoNode<T>>& all_nodes) const;
  template <typename T>
  void setParentId(TopoNode<T>& node, const int& id, std::vector<TopoNode<T>>& all_nodes) const;
  template <typename T>
  void init_node_name(Topology<T>& topo, const std::string& clk_topo_name);
  template <typename T>
  ClockTopo create_clock_topo(Topology<T>& topo, const std::string& clk_topo_name);
  ClockTopo create_clock_topo(CtsNet* clk_net);
  std::string connect_string(const std::string& net_name, int level, int index) const;

  std::vector<CtsInstance*> get_clustering_insts(CtsNet* clk_net);

  std::vector<CtsClock*> _clocks;
  std::vector<ClockTopo> _clk_topos;
  int _end_index;
  int _steiner_index = 0;
  std::unordered_map<std::string, CtsInstance*> _name_to_inst;
  SkewScheduler* _skew_scheduler;
};

}  // namespace icts