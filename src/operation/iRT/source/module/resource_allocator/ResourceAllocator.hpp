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
#include "DataManager.hpp"
#include "Database.hpp"
#include "GridMap.hpp"
#include "RAGCell.hpp"
#include "RAModel.hpp"
#include "RANet.hpp"

namespace irt {

#define RA_INST (irt::ResourceAllocator::getInst())

class ResourceAllocator
{
 public:
  static void initInst();
  static ResourceAllocator& getInst();
  static void destroyInst();
  // function
  void allocate(std::vector<Net>& net_list);

 private:
  // self
  static ResourceAllocator* _ra_instance;

  ResourceAllocator() = default;
  ResourceAllocator(const ResourceAllocator& other) = delete;
  ResourceAllocator(ResourceAllocator&& other) = delete;
  ~ResourceAllocator() = default;
  ResourceAllocator& operator=(const ResourceAllocator& other) = delete;
  ResourceAllocator& operator=(ResourceAllocator&& other) = delete;
  // function
  void allocateNetList(std::vector<Net>& net_list);

#if 1  // build ra_model
  RAModel initRAModel(std::vector<Net>& net_list);
  std::vector<RANet> convertToRANetList(std::vector<Net>& net_list);
  RANet convertToRANet(Net& net);
  void buildRAModel(RAModel& ra_model);
  void initRANetDemand(RAModel& ra_model);
  void initRAGCellList(RAModel& ra_model);
  void updateNetBlockageMap(RAModel& ra_model);
  void addRectToEnv(RAModel& ra_model, irt_int net_idx, LayerRect real_rect);
  void cutBlockageList(RAModel& ra_model);
  void buildAccessMap(RAModel& ra_model);
  void calcRAGCellSupply(RAModel& ra_model);
  std::vector<PlanarRect> getWireList(RAGCell& ra_gcell, RoutingLayer& routing_layer);
  void buildRelation(RAModel& ra_model);
  void initTempObject(RAModel& ra_model);
#endif

#if 1  // check ra_model
  void checkRAModel(RAModel& ra_model);
#endif

#if 1  // allocate ra_model
  void allocateRAModel(RAModel& ra_model);
  void calcNablaF(RAModel& ra_model, double penalty_para);
  double calcAlpha(RAModel& ra_model, double penalty_para);
  double updateResult(RAModel& ra_model);
#endif

#if 1  // update ra_model
  void updateRAModel(RAModel& ra_model);
  void updateAllocationMap(RAModel& ra_model);
  GridMap<double> getCostMap(GridMap<double>& allocation_map, double lower_cost);
  void normalizeCostMap(GridMap<double>& cost_map, double lower_cost);
  void updateOriginRACostMap(RAModel& ra_model);
#endif

#if 1  // report ra_model
  void reportRAModel(RAModel& ra_model);
  void countRAModel(RAModel& ra_model);
  void reportTable(RAModel& ra_model);
#endif
};

}  // namespace irt
