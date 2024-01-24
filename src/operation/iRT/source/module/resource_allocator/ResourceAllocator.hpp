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
  void writePYScript();
  RAModel initRAModel(std::vector<Net>& net_list);
  std::vector<RANet> convertToRANetList(std::vector<Net>& net_list);
  RANet convertToRANet(Net& net);
  void initRANetDemand(RAModel& ra_model);
  void initRAGCellList(RAModel& ra_model);
  void calcRAGCellSupply(RAModel& ra_model);
  std::vector<PlanarRect> getWireList(RAGCell& ra_gcell, RoutingLayer& routing_layer);
  void buildRelation(RAModel& ra_model);
  void initTempObject(RAModel& ra_model);
  void checkRAModel(RAModel& ra_model);
  void iterativeRAModel(RAModel& ra_model);
  /**
   * @description: 使用二次规划 罚方法
   *
   * 迭代过程
   *  f(x) = (1/2) * x' * Q * x + b' * x
   *  nabla_f = Q * x + b
   *  alpha = (nabla_f' * nabla_f) / (nabla_f' * Q * nabla_f)
   *  x = x + (-nabla_f) * alpha
   *
   * eg.
   *  RAGCell : GCM    Target : T    RANet : NM    Constraint : C
   * ────────────────────────────────────────
   * │ GCM0(T1=3) │ GCM1(T2=4) │ GCM2(T3=5) │
   * ───────────────────────────────────────────────────│
   * │     x0     │     x1     │            │ NM0(C1=3) │
   * │            │            │            │           │
   * │            │     x2     │     x3     │ NM1(C2=6) │
   * ───────────────────────────────────────────────────│
   *
   * min [(x0 - T1)^2 + (x1 + x2 - T2)^2 + (x3 - T3)^2] +
   *     (1/(2 * u)) * [(x0 + x1 - C1)^2 + (x2 + x3 - C2)^2]
   *
   */
  void allocateRAModel(RAModel& ra_model, double penalty_para);
  /**
   * calculate nabla_f;
   * 因;
   * nabla_f = Q * x + b
   * 且;
   * Q = Q_ra_gcell + (1/2u) * Q_ra_net;
   * F = F_ra_gcell + (1/2u) * F_ra_net;
   * 代入;
   * nabla_f = Q_ra_gcell'*X + (1/2u)Q_ra_net'*X + F_ra_gcell + (1/2u)F_ra_net
   * 整理;
   * nabla_f = (Q_ra_gcell*X + F_ra_gcell) + (1/2u)*(Q_ra_net*X + F_ra_net)
   */
  void calcNablaF(RAModel& ra_model, double penalty_para);
  //  calculate alpha = (nabla_f' * nabla_f) / (nabla_f' * Q * nabla_f)
  double calcAlpha(RAModel& ra_model, double penalty_para);
  // x = x + (-nabla_f) * alpha
  double updateResult(RAModel& ra_model);
  void processRAModel(RAModel& ra_model);
  GridMap<double> getCostMap(GridMap<double>& allocation_map, double lower_cost);
  void normalizeCostMap(GridMap<double>& cost_map, double lower_cost);
  void update(RAModel& ra_model);
#if 1  // plot ra_model
  void outputResourceMap(RAModel& ra_model);
#endif
};

}  // namespace irt
