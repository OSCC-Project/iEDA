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

#include "Config.hpp"
#include "EGRDataManager.hpp"
#include "EGRRoutingPackage.hpp"
#include "GDSPlotter.hpp"
#include "flute3/flute.h"
#include "libfort/fort.hpp"

namespace irt {

#define EGR_INST (irt::EarlyGlobalRouter::getInst())
class EarlyGlobalRouter
{
 public:
  static void initInst(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  static EarlyGlobalRouter& getInst();
  static void destroyInst();
  // function
  void route();
  void recordLog(std::string record_file_path);

  void plot();
  void plotCongstLoc();
  EGRDataManager& getDataManager() { return _egr_data_manager; }

 private:
  // self
  static EarlyGlobalRouter* _egr_instance;
  // config & database
  EGRDataManager _egr_data_manager;
  idb::IdbBuilder* _idb_builder;

  EarlyGlobalRouter(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder) { init(config_map, idb_builder); }
  EarlyGlobalRouter(const EarlyGlobalRouter& other) = delete;
  EarlyGlobalRouter(EarlyGlobalRouter&& other) = delete;
  ~EarlyGlobalRouter() { destroy(); }
  EarlyGlobalRouter& operator=(const EarlyGlobalRouter& other) = delete;
  EarlyGlobalRouter& operator=(EarlyGlobalRouter&& other) = delete;
  // function
  void init(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  void destroy();
  void routeEGRNetList(std::vector<EGRNet>& egr_net_list);
  void routeEGRNet(EGRNet& egr_net);
  bool skipRouting(EGRNet& egr_net);
  EGRRoutingPackage initEGRRoutingPackage(EGRNet& egr_net);
  void routeEGRRoutingPackage(EGRRoutingPackage& egr_routing_package);
  void updateNearestCoordPair(EGRRoutingPackage& egr_routing_package);
  void routeNearestCoordPair(EGRRoutingPackage& egr_routing_package);
  void routeByGradual(EGRRoutingPackage& egr_routing_package);
  void routeByTopo(EGRRoutingPackage& egr_routing_package);
  void generateFluteTree(EGRRoutingPackage& egr_routing_package);
  void generateCoordPairList(EGRRoutingPackage& egr_routing_package);
  irt_int getMinCostLayerIdx(const PlanarCoord& planar_coord);
  LayerCoord getNearestCoordOnSegment(LayerCoord& coord, Segment<LayerCoord>& segment);
  void routeAllCoordPairs(EGRRoutingPackage& egr_routing_package);
  void initTempData(EGRRoutingPackage& egr_routing_package);
  std::vector<Segment<LayerCoord>> routeInPattern(std::pair<LayerCoord, LayerCoord>& coord_pair);
  bool updateBestSegmentList(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                             std::vector<Segment<LayerCoord>>& best_routing_segment_list, double& best_path_cost);
  void routeByStraight(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                       std::pair<LayerCoord, LayerCoord>& coord_pair);
  void routeByLPattern(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                       std::pair<LayerCoord, LayerCoord>& coord_pair);
  void routeByUPattern(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                       std::pair<LayerCoord, LayerCoord>& coord_pair);
  void routeByZPattern(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                       std::pair<LayerCoord, LayerCoord>& coord_pair);
  std::vector<irt_int> getMidIndexList(irt_int start_idx, irt_int end_idx);
  void routeByInner3BendsPattern(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                                 std::pair<LayerCoord, LayerCoord>& coord_pair);
  void routeByOuter3BendsPattern(std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                                 std::pair<LayerCoord, LayerCoord>& coord_pair);
  void updateRoutingSegmentList(EGRNet& egr_net, EGRRoutingPackage& egr_routing_package);
  void updateLayerResourceMap(EGRNet& egr_net);
  void addDemandBySegmentList(std::vector<Segment<TNode<LayerCoord>*>>& segment_list);
  // report
  void reportEGRNetList();
  void calcuResult();
  void calcuCongestion();
  void calcuWireViaStatistics();
  void reportResult();
  void reportCongestion();
  void reportWireViaStatistics();
};
}  // namespace irt
