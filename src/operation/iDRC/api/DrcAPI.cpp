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
#include "DrcAPI.hpp"

#include "CornerFillSpacingCheck.hpp"
#include "CutEolSpacingCheck.hpp"
#include "CutSpacingCheck.hpp"
#include "DrcIDBWrapper.h"
#include "EOLSpacingCheck.hpp"
#include "EnclosedAreaCheck.h"
#include "EnclosureCheck.hpp"
#include "JogSpacingCheck.hpp"
#include "MinStepCheck.hpp"
#include "NotchSpacingCheck.hpp"
#include "RoutingAreaCheck.h"
#include "RoutingSpacingCheck.h"
#include "RoutingWidthCheck.h"
#include "idm.h"

using namespace std;

namespace idrc {

DrcAPI* DrcAPI::_drc_api_instance = nullptr;

void DrcAPI::destroyInst()
{
  if (_drc_api_instance != nullptr) {
    delete _drc_api_instance;
    _drc_api_instance = nullptr;
  }
}
// function

void DrcAPI::initDRC()
{
  delete _tech;
  _tech = new Tech();
  if (dmInst->get_idb_builder()) {
    _idb_wrapper = new DrcIDBWrapper(_tech, dmInst->get_idb_builder());
    _idb_wrapper->wrapTech();
  } else {
    std::cout << "Error: idb builder is null" << std::endl;
    exit(1);
  }
}

RegionQuery* DrcAPI::init()
{
  RegionQuery* region_query = new RegionQuery(_tech);
  return region_query;
}

void DrcAPI::destroy(RegionQuery* region_query)
{
}

std::map<std::string, int> DrcAPI::getCheckResult()
{
  runDrc();
  return DrcInst.getDrcResult();
}

void DrcAPI::runDrc()
{
  DrcInst.initDRC();
  DrcInst.initCheckModule();
  DrcInst.run();
}

std::map<std::string, std::vector<DrcViolationSpot*>> DrcAPI::getDetailCheckResult()
{
  //   DrcInst.initDRC();
  //   DrcInst.initCheckModule();
  //   DrcInst.run();
  return DrcInst.getDrcDetailResult();
}

std::map<std::string, int> DrcAPI::getCheckResult(RegionQuery* region_query)
{
  RoutingSpacingCheck* routing_spacing_check = new RoutingSpacingCheck(_tech, region_query);
  EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, region_query);
  NotchSpacingCheck* notch_spacing_check = new NotchSpacingCheck(_tech, region_query);
  RoutingWidthCheck* width_check = new RoutingWidthCheck(_tech, region_query);
  MinStepCheck* min_step_check = new MinStepCheck(_tech, region_query);
  RoutingAreaCheck* area_check = new RoutingAreaCheck(_tech, region_query);
  CornerFillSpacingCheck* corner_fill_spacing_check = new CornerFillSpacingCheck(_tech, region_query);
  CutSpacingCheck* cut_spacing_check = new CutSpacingCheck(_tech, region_query);
  CutEolSpacingCheck* cut_eol_spacing_check = new CutEolSpacingCheck(_tech, region_query);
  EnclosureCheck* enclosure_check = new EnclosureCheck(_tech, region_query);
  // EnclosedAreaCheck* enclosed_area_check = new EnclosedAreaCheck(_tech, region_query);
  auto routing_rect_set = region_query->getRoutingRectSet();
  auto cut_rect_set = region_query->getCutRectSet();
  for (auto routing_rect : routing_rect_set) {
    routing_spacing_check->check(routing_rect);
    width_check->check(routing_rect);
  }
  for (auto cut_rect : cut_rect_set) {
    cut_eol_spacing_check->check(cut_rect);
    cut_spacing_check->check(cut_rect);
    enclosure_check->check(cut_rect);
  }

  auto layer_id_and_net_id_to_polys_map = region_query->getRegionPolysMap();
  for (auto& [layer_id, net_id_to_polys_map] : layer_id_and_net_id_to_polys_map) {
    for (auto [net_id, polys_set] : net_id_to_polys_map) {
      for (auto& poly : polys_set) {
        eol_spacing_check->check(poly);
        notch_spacing_check->check(poly);
        min_step_check->check(poly);
        area_check->check(poly);
        corner_fill_spacing_check->check(poly);
      }
    }
  }
  std::map<std::string, int> viotype_to_nums_map;
  region_query->getRegionReport(viotype_to_nums_map);
  return viotype_to_nums_map;
}

// void DrcAPI::add(std::vector<ids::DRCTask> task_list)
// {
//   for (auto& [region_query, drc_rect_list] : task_list) {
//     std::set<DrcPoly*> intersect_poly_set;
//     // 得到与这组rect相交的所有polygon
//     region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
//     // 删除与这组rect相交的所有polygon
//     region_query->deleteIntersectPoly(intersect_poly_set);

//     DrcPoly* new_poly = region_query->rebuildPoly_add(intersect_poly_set, drc_rect_list);
//     if (new_poly) {
//       region_query->addPoly(new_poly);
//     }
//     for (auto& drc_rect : drc_rect_list) {
//       region_query->addDrcRect(drc_rect, _tech);
//     }
//   }
// }

void DrcAPI::add(RegionQuery* region_query, std::vector<idrc::DrcRect*> drc_rect_list)
{
  std::vector<DrcRect*>& region_rect_list = region_query->getRegionRectList();
  region_rect_list.insert(region_rect_list.end(), drc_rect_list.begin(), drc_rect_list.end());

  std::set<DrcPoly*> intersect_poly_set;
  // 得到与这组rect相交的所有polygon
  region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
  // 删除与这组rect相交的所有polygon
  region_query->deleteIntersectPoly(intersect_poly_set);

  //   DrcPoly* new_poly = region_query->rebuildPoly_add(intersect_poly_set, drc_rect_list);
  //   if (new_poly) {
  //     region_query->addPoly(new_poly);
  //   }
  //   for (auto& drc_rect : drc_rect_list) {
  //     region_query->addDrcRect(drc_rect, _tech);
  //   }

  auto new_poly_list = region_query->rebuildPoly_add_list(intersect_poly_set, drc_rect_list);
  for (auto new_poly : new_poly_list) {
    if (new_poly) {
      region_query->addPoly(new_poly);
    }
    for (auto& drc_rect : drc_rect_list) {
      region_query->addDrcRect(drc_rect, _tech);
    }
  }
}

// void DrcAPI::mergeRectToPoly(DrcRect* check_rect, RegionQuery* region_query, DrcPoly* poly)
// {
//   std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
//   int layer_id = check_rect->get_layer_id();
//   region_query->queryInRoutingLayer(layer_id, DRCUtil::getRTreeBox(check_rect), query_result);
// }

bool DrcAPI::checkSpacing_rect(DrcRect* check_rect, RegionQuery* region_query)
{
  RoutingSpacingCheck* routing_spacing_check = new RoutingSpacingCheck(_tech, region_query);
  if (!routing_spacing_check->check(check_rect)) {
    delete routing_spacing_check;
    return false;
  }
  std::set<DrcPoly*> intersect_poly_set;
  std::vector<DrcRect*> drc_rect_list;
  drc_rect_list.push_back(check_rect);
  region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
  DrcPoly* poly = region_query->rebuildPoly_add(intersect_poly_set, drc_rect_list);

  /// 无法构建成 1 个poly，无法检测，暂时忽略处理
  if (poly == nullptr) {
    return true;
  }

  // 给poly build edge
  region_query->addPolyEdge_NotAddToRegion(poly);
  EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, region_query);
  if (!eol_spacing_check->check(poly)) {
    delete eol_spacing_check;
    return false;
  }

  CornerFillSpacingCheck* corner_fill_check = new CornerFillSpacingCheck(_tech, region_query);
  if (!corner_fill_check->check(poly)) {
    delete corner_fill_check;
    return false;
  }
  delete routing_spacing_check;
  delete eol_spacing_check;
  delete corner_fill_check;
  return true;
}

bool DrcAPI::checkSpacing(RegionQuery* region_query, DrcRect* check_rect)
{
  std::vector<std::pair<RTreeBox, DrcRect*>> rect_query_result;
  std::vector<std::pair<RTreeSegment, DrcEdge*>> edge_query_result;
  int layer_id = check_rect->get_layer_id();

  region_query->queryEdgeInRoutingLayer(layer_id, DRCUtil::getRTreeBox(check_rect), edge_query_result);
  region_query->queryInRoutingLayer(layer_id, DRCUtil::getRTreeBox(check_rect), rect_query_result);
  std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>> max_scope_query_result;
  region_query->queryInMaxScope(layer_id, DRCUtil::getRTreeBox(check_rect), max_scope_query_result);
  std::map<void*, std::map<ScopeType, std::vector<DrcRect*>>> min_scope_query_result;
  region_query->queryInMinScope(layer_id, DRCUtil::getRTreeBox(check_rect), min_scope_query_result);

  // 排除与poly相交，rect直接query 出edge和rect，移除掉同一net的，不同net的直接报短路违例，再把这些edge、rect从min和max中移除
  for (auto [rtree_seg, edge] : edge_query_result) {
    min_scope_query_result.erase(edge);
    max_scope_query_result.erase(edge);
    if (edge->get_owner_polygon()->getNetId() != check_rect->get_net_id()) {
      return false;
    }
  }
  for (auto [rtree_box, rect] : rect_query_result) {
    min_scope_query_result.erase(rect);
    max_scope_query_result.erase(rect);
    if (rect->get_net_id() != check_rect->get_net_id()) {
      return false;
    }
  }
  bool intersect_with_min_scope = !min_scope_query_result.empty();
  bool intersect_with_max_scope = !max_scope_query_result.empty();
  if (intersect_with_min_scope) {
    return false;
  } else if (intersect_with_max_scope) {
    for (auto [target, scope_type_set] : max_scope_query_result) {
      if (scope_type_set.contains(ScopeType::EOL)) {
        EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, region_query);
        if (!eol_spacing_check->check(target, check_rect)) {
          delete eol_spacing_check;
          return false;
        }
        delete eol_spacing_check;
      }
      if (scope_type_set.contains(ScopeType::CornerFill)) {
        return false;
      }
      if (scope_type_set.contains(ScopeType::Common)) {
        RoutingSpacingCheck* routing_spacing_check = new RoutingSpacingCheck(_tech, region_query);
        if (!routing_spacing_check->check(target, check_rect)) {
          delete routing_spacing_check;
          return false;
        }
        delete routing_spacing_check;
      }
    }
  } else {
    // 以rect为主体查，融合为poly，过程同读def的检查流程
    return checkSpacing_rect(check_rect, region_query);
  }
  return true;
}

bool DrcAPI::checkShape(RegionQuery* region_query, DrcRect* check_rect)
{
  RoutingWidthCheck* width_check = new RoutingWidthCheck(_tech, region_query);
  if (!width_check->check(check_rect)) {
    delete width_check;
    return false;
  }
  std::set<DrcPoly*> intersect_poly_set;
  std::vector<DrcRect*> drc_rect_list;
  drc_rect_list.push_back(check_rect);
  region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
  DrcPoly* poly = region_query->rebuildPoly_add(intersect_poly_set, drc_rect_list);
  NotchSpacingCheck* notch_spacing_check = new NotchSpacingCheck(_tech, region_query);
  if (!notch_spacing_check->check(poly)) {
    delete notch_spacing_check;
    return false;
  }

  MinStepCheck* min_step_check = new MinStepCheck(_tech, region_query);
  if (!min_step_check->check(poly)) {
    delete min_step_check;
    return false;
  }

  RoutingAreaCheck* area_check = new RoutingAreaCheck(_tech, region_query);
  if (!area_check->check(poly)) {
    delete area_check;
    return false;
  }
  delete width_check;
  delete notch_spacing_check;
  delete min_step_check;
  delete area_check;
  return true;
}

// bool DrcAPI::check(std::vector<ids::DRCTask> task_list)
// {
//   for (auto& [region_query, drc_rect_list] : task_list) {
//     for (auto& drc_rect : drc_rect_list) {
//       if (!checkSpacing(region_query, drc_rect)) {
//         return false;
//       }

//       if (!checkShape(region_query, drc_rect)) {
//         return false;
//       }

//       // checkMinimumCut(drc_rect);
//     }
//   }
//   return true;
// }

bool DrcAPI::check(RegionQuery* region_query, std::vector<idrc::DrcRect*> drc_rect_list)
{
  for (auto& drc_rect : drc_rect_list) {
    if (!checkSpacing(region_query, drc_rect)) {
      return false;
    }

    if (!checkShape(region_query, drc_rect)) {
      return false;
    }

    // checkMinimumCut(drc_rect);
  }

  return true;
}

// void DrcAPI::del(std::vector<ids::DRCTask> task_list)
// {
//   for (auto& [region_query, drc_rect_list] : task_list) {
//     std::set<DrcPoly*> intersect_poly_set;
//     region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
//     region_query->deleteIntersectPoly(intersect_poly_set);
//     auto new_poly_list = region_query->rebuildPoly_del(intersect_poly_set, drc_rect_list);
//     if (!new_poly_list.empty()) {
//       region_query->addPolyList(new_poly_list);
//     }
//     for (auto& drc_rect : drc_rect_list) {
//       region_query->removeDrcRect(drc_rect);
//     }
//   }
// }

void DrcAPI::del(RegionQuery* region_query, std::vector<idrc::DrcRect*> drc_rect_list)
{
  // std::set<DrcPoly*> intersect_poly_set;
  // region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);
  // region_query->deleteIntersectPoly(intersect_poly_set);
  // auto new_poly_list = region_query->rebuildPoly_del(intersect_poly_set, drc_rect_list);
  // if (!new_poly_list.empty()) {
  //   region_query->addPolyList(new_poly_list);
  // }
  // for (auto& drc_rect : drc_rect_list) {
  //   region_query->removeDrcRect(drc_rect);
  // }
  std::set<DrcPoly*> intersect_poly_set;
  region_query->getIntersectPoly(intersect_poly_set, drc_rect_list);

  for (auto& drc_rect : drc_rect_list) {
    region_query->removeDrcRect(drc_rect);
  }

  if (!intersect_poly_set.empty()) {
    std::vector<DrcPoly*> intersect_poly_list;
    intersect_poly_list.assign(intersect_poly_set.begin(), intersect_poly_set.end());
    region_query->addPolyList(intersect_poly_list);
  }
}

// RegionQuery* DrcAPI::createRTree(ids::DRCEnv env)
// {
//   RegionQuery* region_query = new RegionQuery();
//   initRegionQuery(env, region_query);
//   return region_query;
// }

void DrcAPI::initNets(std::vector<DrcRect*>& drc_rect_list, std::map<int, DrcNet>& nets)
{
  for (auto& target_drc_rect : drc_rect_list) {
    int net_id = target_drc_rect->get_net_id();
    int layer_id = target_drc_rect->get_layer_id();
    nets[net_id].set_net_id(net_id);
    if (target_drc_rect->get_owner_type() == RectOwnerType::kViaCut) {
      nets[net_id].add_cut_rect(layer_id, target_drc_rect);
    } else {
      if (target_drc_rect->get_owner_type() == RectOwnerType::kBlockage) {
        continue;
      } else {
        nets[net_id].add_routing_rect(layer_id, target_drc_rect);
        int lb_x = target_drc_rect->get_left();
        int lb_y = target_drc_rect->get_bottom();
        int rt_x = target_drc_rect->get_right();
        int rt_y = target_drc_rect->get_top();
        BoostRect boost_rect(lb_x, lb_y, rt_x, rt_y);
        nets[net_id].add_routing_rect(layer_id, boost_rect);
      }
    }
  }
}

void DrcAPI::initPoly(std::map<int, DrcNet>& nets, RegionQuery* region_query)
{
  for (auto& net_pair : nets) {
    initPolyPolygon(&net_pair.second);
    initPolyEdges(&net_pair.second, region_query);
    // initPolyCorners(&net);
  }
}

// bool DrcAPI::checkDRC(std::vector<DrcRect*> origin_rect_list)
// {
//   // init RectRTreeMap
//   // LayerIdToRTreeMap* region_query = new LayerIdToRTreeMap();
//   RegionQuery* region_query = new RegionQuery();

//   initRegionQuery(origin_rect_list, region_query);
//   std::map<int, DrcNet> nets;
//   initNets(origin_rect_list, nets);
//   initPoly(nets, region_query);
//   // TODO EdgeRtreeMap

//   // init check module
//   // RoutingSpacingCheck* routing_spacing_check = new RoutingSpacingCheck(_tech, region_query);
//   // EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, region_query);
//   // NotchSpacingCheck* notch_spacing_check = new NotchSpacingCheck(_tech, region_query);
//   // RoutingWidthCheck* width_check = new RoutingWidthCheck(_tech, region_query);
//   // MinStepCheck* min_step_check = new MinStepCheck(_tech);
//   // RoutingAreaCheck* area_check = new RoutingAreaCheck(_tech);
//   // CornerFillSpacingCheck* corner_fill_spacing_check = new CornerFillSpacingCheck(_tech, region_query);
//   // CutSpacingCheck* cut_spacing_check = new CutSpacingCheck(_tech, region_query);
//   // CutEolSpacingCheck* cut_eol_spacing_check = new CutEolSpacingCheck(_tech, region_query);
//   // EnclosureCheck* enclosure_check = new EnclosureCheck(_tech, region_query);

//   bool res = true;

//   for (auto& target_drc_rect : origin_rect_list) {
//     if (!cut_spacing_check->check(target_drc_rect)) {
//       return false;
//     }
//   }
//   // for (auto& net : nets) {
//   //   if (!corner_fill_spacing_check->check(&net.second)) {
//   //     res = false;
//   //     break;
//   //   }
//   //   if (!min_step_check->check(&net.second)) {
//   //     res = false;
//   //     break;
//   //   }
//   //   if (!eol_spacing_check->check(&net.second)) {
//   //     res = false;
//   //     break;
//   //   }
//   //   if (!notch_spacing_check->check(&net.second)) {
//   //     res = false;
//   //     break;
//   //   }
//   //   if (!width_check->check(&net.second)) {
//   //     res = false;
//   //     break;
//   //   }
//   // }

//   // delete region_query;
//   // // delete area_check;
//   // // delete min_step_check;
//   // // delete routing_spacing_check;
//   // // delete eol_spacing_check;
//   // // delete corner_fill_spacing_check;

//   return res;
// }

RTreeBox DrcAPI::getRTreeBox(DrcRect* rect)
{
  RTreePoint leftBottom(rect->get_left(), rect->get_bottom());
  RTreePoint rightTop(rect->get_right(), rect->get_top());
  return RTreeBox(leftBottom, rightTop);
}

// void DrcAPI::initSpacingRegion(ids::DRCEnv env, RegionQuery* region_query)
// {
//   for (auto& env_rect : env) {
//     DrcRect* drc_rect = getDrcRect(env_rect);
//     int layer_id = drc_rect->get_layer_id();
//     if (env_rect.type == ids::RectType::kCut) {
//       region_query->add_cut_rect_to_rtree(layer_id, drc_rect);
//     } else {
//       // add to origin rtree
//       region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
//       // add common spacing region to region rtree
//       addCommonSpacingRegion(drc_rect, region_query);
//     }
//   }
//   // add EOL and Corner_Fill Spacing region to region rtree
//   for (auto& [layer_id, net] : region_query->get_nets_map()) {
//     for (auto& [layer_id, target_polys] : net.get_route_polys_list()) {
//       for (auto& target_poly : target_polys) {
//         addEOLSpacingRegion(target_poly.get(), region_query);
//         addCornerFillSpacingRegion();
//         removeEOLSpacingRegion();
//         removeCornerFillSpacingRegion();
//       }
//     }
//   }
// }

// void DrcAPI::initRegionQuery(ids::DRCEnv env, RegionQuery* region_query)
// {
//   initNets(env, region_query->get_nets_map());
//   initPoly(region_query->get_nets_map(), region_query);
//   initSpacingRegion(env, region_query);
// }

// void DrcAPI::getCommonSpacingMinRegion(DrcRect* common_spacing_min_region, DrcRect* drc_rect)
// {
//   int layer_id = drc_rect->get_layer_id();
//   int net_id = drc_rect->get_net_id();
//   int spacing = _tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(0, 0);
//   int lb_x = drc_rect->get_left() - spacing;
//   int lb_y = drc_rect->get_bottom() - spacing;
//   int rt_x = drc_rect->get_right() + spacing;
//   int rt_y = drc_rect->get_top() + spacing;
//   common_spacing_min_region->set_coordinate(lb_x, lb_y, rt_x, rt_y);
//   common_spacing_min_region->set_owner_type(RectOwnerType::kCommonRegion);
//   common_spacing_min_region->set_net_id(net_id);
//   common_spacing_min_region->set_connection(drc_rect);
//   drc_rect->set_connection(common_spacing_min_region);
// }

// void DrcAPI::getCommonSpacingMaxRegion(DrcRect* common_spacing_min_region, DrcRect* drc_rect)
// {
//   int layer_id = drc_rect->get_layer_id();
//   int net_id = drc_rect->get_net_id();
//   int width = drc_rect->getWidth();
//   int length = drc_rect->getLength();
//   int spacing = _tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(width, length);
//   int lb_x = drc_rect->get_left() - spacing;
//   int lb_y = drc_rect->get_bottom() - spacing;
//   int rt_x = drc_rect->get_right() + spacing;
//   int rt_y = drc_rect->get_top() + spacing;
//   common_spacing_min_region->set_coordinate(lb_x, lb_y, rt_x, rt_y);
//   common_spacing_min_region->set_owner_type(RectOwnerType::kCommonRegion);
//   common_spacing_min_region->set_net_id(net_id);
//   common_spacing_min_region->set_connection(drc_rect);
//   drc_rect->set_connection(common_spacing_min_region);
// }

// void DrcAPI::addCommonSpacingRegion(DrcRect* drc_rect, RegionQuery* region_query)
// {
//   DrcRect* common_spacing_min_region = new DrcRect();
//   getCommonSpacingMinRegion(common_spacing_min_region, drc_rect);
//   region_query->add_spacing_min_region(common_spacing_min_region);
//   DrcRect* common_spacing_max_region = new DrcRect();
//   getCommonSpacingMaxRegion(common_spacing_max_region, drc_rect);
//   region_query->add_spacing_max_region(common_spacing_max_region);
// }

// void DrcAPI::getEOLSpacingMinRegion(std::vector<DrcRect*> eol_spacing_min_region_list, DrcPoly* drc_poly)
// {
// }

// void DrcAPI::addEOLSpacingRegion(DrcPoly* drc_poly, RegionQuery* region_query)
// {
//   std::vector<DrcRect*> eol_spacing_min_region_list;
//   getEOLSpacingMinRegion(eol_spacing_min_region_list, drc_poly);
//   region_query->add_spacing_min_region(eol_spacing_min_region_list);
//   std::vector<DrcRect*> eol_spacing_max_region_list;
//   getEOLSpacingMaxRegion(eol_spacing_max_region_list, drc_poly);
//   region_query->add_spacing_max_region(eol_spacing_max_region_list);
// }

void DrcAPI::initRegionQuery(std::vector<DrcRect*> origin_rect_list, RegionQuery* region_query)
{
  for (auto& drc_rect : origin_rect_list) {
    int layer_id = drc_rect->get_layer_id();
    if (drc_rect->get_owner_type() == RectOwnerType::kViaCut) {
      region_query->add_cut_rect_to_rtree(layer_id, drc_rect);
    } else {
      region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
    }
  }
}

void DrcAPI::initPolyEdges(DrcNet* net, RegionQuery* region_query)
{
  int routing_layer_num = _tech->get_drc_routing_layer_list().size();
  // test_eol
  // int routing_layer_num = 1;
  // std::vector<std::set<std::pair<DrcCoordinate<int>, DrcCoordinate<int>>>> polygons_edges(routing_layer_num);
  // Assign the edges of the fused polygon to the edges in the poly
  for (int i = 0; i < routing_layer_num; i++) {
    for (auto& poly : net->get_route_polys(i)) {
      auto polygon = poly->getPolygon();
      initPolyOuterEdges(net, poly.get(), polygon, i, region_query);
      // pending
      for (auto holeIt = polygon->begin_holes(); holeIt != polygon->end_holes(); holeIt++) {
        auto& hole_poly = *holeIt;
        initPolyInnerEdges(net, poly.get(), hole_poly, i, region_query);
      }
    }
  }
}

void DrcAPI::initPolyOuterEdges(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id, RegionQuery* region_query)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto outerIt = polygon->begin();
  bp.set((*outerIt).x(), (*outerIt).y());

  bp1 = *outerIt;
  firstPt.set((*outerIt).x(), (*outerIt).y());
  firstPt1 = *outerIt;
  outerIt++;
  // loop from second to last pt (n-1) edges
  for (; outerIt != polygon->end(); outerIt++) {
    ep.set((*outerIt).x(), (*outerIt).y());
    ep1 = *outerIt;
    // auto edge = make_unique<DrcEdge>();
    std::unique_ptr<DrcEdge> edge(new DrcEdge);
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    // edge->setPoints(bp, ep);
    edge->setSegment(bp1, ep1);
    edge->setDir();
    edge->set_is_fixed(false);
    if (region_query) {
      region_query->add_routing_edge_to_rtree(layer_id, edge.get());
    }
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  // last edge
  auto edge = make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  // edge->setPoints(bp, firstPt);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());
  if (region_query) {
    region_query->add_routing_edge_to_rtree(layer_id, edge.get());
  }
  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void DrcAPI::initPolyInnerEdges(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id, RegionQuery* region_query)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  vector<unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto innerIt = hole_poly.begin();
  bp.set((*innerIt).x(), (*innerIt).y());
  bp1 = *innerIt;
  firstPt.set((*innerIt).x(), (*innerIt).y());
  firstPt1 = *innerIt;
  innerIt++;
  // loop from second to last pt (n-1) edges
  for (; innerIt != hole_poly.end(); innerIt++) {
    ep.set((*innerIt).x(), (*innerIt).y());
    ep1 = *innerIt;
    auto edge = make_unique<DrcEdge>();
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    edge->setDir();
    edge->setSegment(bp1, ep1);
    if (region_query) {
      region_query->add_routing_edge_to_rtree(layer_id, edge.get());
    }
    edge->set_is_fixed(false);
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  auto edge = make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  if (region_query) {
    region_query->add_routing_edge_to_rtree(layer_id, edge.get());
  }
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

/**
 * @brief Assign each layer of net polygon to each poly
 *
 * @param net
 */
void DrcAPI::initPolyPolygon(DrcNet* net)
{
  int routing_layer_num = _tech->get_drc_routing_layer_list().size();
  // test_eol
  // int routing_layer_num = 1;

  std::vector<PolygonSet> layer_routing_polys(routing_layer_num);
  std::vector<PolygonWithHoles> polygons;

  for (int routing_layer_id = 0; routing_layer_id < routing_layer_num; routing_layer_id++) {
    polygons.clear();
    layer_routing_polys[routing_layer_id] = net->get_routing_polygon_set_by_id(routing_layer_id);
    // 输出到polys中；
    layer_routing_polys[routing_layer_id].get(polygons);
    for (auto& polygon : polygons) {
      net->addPoly(polygon, routing_layer_id);
    }
  }
}

void DrcAPI::getCommonSpacingScopeRect(DrcRect* target_rect, DrcRect* scope_rect, int spacing)
{
  int lb_x = target_rect->get_left() - spacing;
  int lb_y = target_rect->get_bottom() - spacing;
  int rt_x = target_rect->get_right() + spacing;
  int rt_y = target_rect->get_top() + spacing;
  scope_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  scope_rect->set_layer_id(target_rect->get_layer_id());
  scope_rect->set_layer_order(target_rect->get_layer_order());
}

void DrcAPI::getCommonSpacingScope(std::vector<DrcRect*>& max_scope_list, std::map<int, DrcNet>& target_nets, bool is_max)
{
  for (auto& [net_id, net] : target_nets) {
    for (auto& [layer_id, routing_rect_list] : net.get_layer_to_routing_rects_map()) {
      for (auto target_rect : routing_rect_list) {
        int spacing;
        if (is_max) {
          int width = target_rect->getWidth();
          int length = target_rect->getLength();
          spacing = _tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(width, length);
        } else {
          spacing = _tech->get_drc_routing_layer_list()[layer_id]->get_spacing_table()->get_parallel()->get_spacing(0, 0);
        }
        DrcRect* common_spacing_max_scope = new DrcRect();
        getCommonSpacingScopeRect(target_rect, common_spacing_max_scope, spacing);
        max_scope_list.push_back(common_spacing_max_scope);
      }
    }
  }
}

void DrcAPI::getEOLSpacingScope(std::vector<DrcRect*>& max_scope_list, std::map<int, DrcNet>& target_nets, bool is_max)
{
  EOLSpacingCheck* eol_spacing_check = new EOLSpacingCheck(_tech, nullptr);
  for (auto& [net_id, target_net] : target_nets) {
    for (auto& [layer_id, target_polys] : target_net.get_route_polys_list()) {
      for (auto& target_poly : target_polys) {
        // getEOLSpacingMaxScopeOfPoly(target_poly.get(), max_scope_list);
        eol_spacing_check->getScope(target_poly.get(), max_scope_list, is_max);
      }
    }
  }
  delete eol_spacing_check;
}

void DrcAPI::getCornerFillSpacingScope(std::vector<DrcRect*>& max_scope_list, std::map<int, DrcNet>& target_nets)
{
  CornerFillSpacingCheck* cornerfill_check = new CornerFillSpacingCheck(_tech, nullptr);
  for (auto& [net_id, target_net] : target_nets) {
    for (auto& [layer_id, target_polys] : target_net.get_route_polys_list()) {
      for (auto& target_poly : target_polys) {
        cornerfill_check->getScope(target_poly.get(), max_scope_list);
      }
    }
  }
  delete cornerfill_check;
}

std::vector<DrcRect*> DrcAPI::getMaxScope(std::vector<DrcRect*> origin_rect_list)
{
  std::vector<DrcRect*> max_scope_list;
  std::map<int, DrcNet> nets;
  initNets(origin_rect_list, nets);
  initPoly(nets, nullptr);
  getCommonSpacingScope(max_scope_list, nets, true);
  // getEOLSpacingScope(max_scope_list, nets, true);
  // getCornerFillSpacingScope(max_scope_list, nets);
  return max_scope_list;
}

std::vector<DrcRect*> DrcAPI::getMinScope(std::vector<DrcRect*> origin_rect_list)
{
  std::vector<DrcRect*> min_scope_list;
  std::map<int, DrcNet> nets;
  initNets(origin_rect_list, nets);
  initPoly(nets, nullptr);
  getCommonSpacingScope(min_scope_list, nets, false);
  // getEOLSpacingScope(min_scope_list, nets, false);
  // getCornerFillSpacingScope(min_scope_list, nets);
  return min_scope_list;
}

DrcRect* DrcAPI::getDrcRect(int net_id, int lb_x, int lb_y, int rt_x, int rt_y, std::string layer_name, bool is_artifical)
{
  DrcRect* drc_rect = new DrcRect(net_id, lb_x, lb_y, rt_x, rt_y);
  if (_tech) {
    int layer_order = _tech->getLayerOrderByName(layer_name);
    drc_rect->set_layer_order(layer_order);
  } else {
    return nullptr;
  }
  std::pair<bool, int> layer_info = _tech->getLayerInfoByLayerName(layer_name);
  drc_rect->set_layer_id(layer_info.second);
  if (is_artifical) {
    drc_rect->set_owner_type(RectOwnerType::kBlockage);
  } else {
    if (!layer_info.first) {
      drc_rect->set_owner_type(RectOwnerType::kViaCut);
    } else {
      drc_rect->set_owner_type(RectOwnerType::kRoutingMetal);
    }
  }
  return drc_rect;
}

DrcRect* DrcAPI::getDrcRect(ids::DRCRect ids_rect)
{
  return getDrcRect(-1, ids_rect.lb_x, ids_rect.lb_y, ids_rect.rt_x, ids_rect.rt_y, ids_rect.layer_name, ids_rect.is_artificial);
}

ids::DRCRect DrcAPI::getDrcRect(DrcRect* drc_rect)
{
  ids::DRCRect ids_rect;
  ids_rect.lb_x = drc_rect->get_rectangle().get_lb_x();
  ids_rect.lb_y = drc_rect->get_rectangle().get_lb_y();
  ids_rect.rt_x = drc_rect->get_rectangle().get_rt_x();
  ids_rect.rt_y = drc_rect->get_rectangle().get_rt_y();
  ids_rect.layer_name = _tech->getRoutingLayerNameById(drc_rect->get_layer_id());
  return ids_rect;
}

std::map<std::string, std::vector<DrcViolationSpot*>> DrcAPI::check(RegionQuery* region_query)
{
  return check(region_query->getRegionRectList());
}

std::map<std::string, std::vector<DrcViolationSpot*>> DrcAPI::check(std::vector<DrcRect*>& region_rect_list, RegionQuery* dr_region_query)
{
  std::vector<DrcRect*> cut_rect_list;
  std::vector<DrcRect*> routing_rect_list;
  std::map<int, DrcNet> nets;

  RegionQuery* region_query = nullptr;
  if (dr_region_query == nullptr) {
    /* init region*/
    region_query = new RegionQuery();
    for (auto& drc_rect : region_rect_list) {
      int layer_id = drc_rect->get_layer_id();
      if (drc_rect->get_owner_type() == RectOwnerType::kViaCut) {
        cut_rect_list.push_back(drc_rect);
        region_query->add_cut_rect_to_rtree(layer_id, drc_rect);
      } else if (drc_rect->get_owner_type() == RectOwnerType::kRoutingMetal) {
        routing_rect_list.push_back(drc_rect);
        region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
      }
    }
  } else {
    region_query = dr_region_query;
  }

  initNets(region_rect_list, nets);
  initPoly(nets, region_query);
  auto jog_spacing_check = new JogSpacingCheck(_tech, region_query);
  auto notch_spacing_check = new NotchSpacingCheck(_tech, region_query);
  auto min_step_check = new MinStepCheck(_tech, region_query);
  auto corner_fill_spacing_check = new CornerFillSpacingCheck(_tech, region_query);
  auto cut_eol_spacing_check = new CutEolSpacingCheck(_tech, region_query);
  auto routing_sapcing_check = new RoutingSpacingCheck(_tech, region_query);
  auto eol_spacing_check = new EOLSpacingCheck(_tech, region_query);
  auto routing_area_check = new RoutingAreaCheck(_tech, region_query);
  auto routing_width_check = new RoutingWidthCheck(_tech, region_query);
  auto enclosed_area_check = new EnclosedAreaCheck(_tech, region_query);
  auto cut_spacing_check = new CutSpacingCheck(_tech, region_query);
  auto enclosure_check = new EnclosureCheck(_tech, region_query);
  for (auto& [net_id, net] : nets) {
    routing_sapcing_check->checkRoutingSpacing(&net);

    routing_width_check->checkRoutingWidth(&net);

    routing_area_check->checkArea(&net);

    enclosed_area_check->checkEnclosedArea(&net);

    cut_spacing_check->checkCutSpacing(&net);

    eol_spacing_check->checkEOLSpacing(&net);

    notch_spacing_check->checkNotchSpacing(&net);

    min_step_check->checkMinStep(&net);

    corner_fill_spacing_check->checkCornerFillSpacing(&net);

    cut_eol_spacing_check->checkCutEolSpacing(&net);

    jog_spacing_check->checkJogSpacing(&net);
  }
  delete jog_spacing_check;
  delete notch_spacing_check;
  delete min_step_check;
  delete corner_fill_spacing_check;
  delete cut_eol_spacing_check;
  delete routing_sapcing_check;
  delete eol_spacing_check;
  delete routing_area_check;
  delete routing_width_check;
  delete enclosed_area_check;
  delete cut_spacing_check;
  delete enclosure_check;

  std::map<std::string, std::vector<DrcViolationSpot*>> vio_map;
  region_query->getRegionDetailReport(vio_map);
  return vio_map;
}

}  // namespace idrc