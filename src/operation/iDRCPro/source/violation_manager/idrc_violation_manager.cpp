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
#include "idrc_violation_manager.h"

#include <boost/polygon/polygon_90_set_concept.hpp>
#include <boost/polygon/rectangle_concept.hpp>
#include <climits>

#include "DRCViolationType.h"
#include "boost_definition.h"
#include "geometry_polygon.h"
#include "geometry_rectangle.h"
#include "idrc_engine_manager.h"
#include "idrc_util.h"
#include "idrc_violation.h"

namespace idrc {

DrcViolationManager::~DrcViolationManager()
{
  for (auto& [type, violations] : _violation_list) {
    for (auto* violation : violations) {
      if (violation != nullptr) {
        delete violation;
        violation = nullptr;
      }
    }

    violations.clear();
    std::vector<DrcViolation*>().swap(violations);
  }
  _violation_list.clear();
}

void DrcViolationManager::set_net_ids(DrcEngineManager* engine_manager)
{
  for (auto& [type, violation_list] : _violation_list) {
    ieda::Stats states;
    std::vector<DrcViolation*> new_violation_list;
    std::string rule_name = idrc::GetViolationTypeName()(type);

    // #pragma omp parallel for
    for (auto* violation : violation_list) {
      if (violation == nullptr) {
        continue;
      }
      // corner
      // FIXME: CORNER TO CORNER PRL BUGS
      if (violation->get_type() == Type::kPolygon) {
        auto* violation_polygon = static_cast<DrcViolationPolygon*>(violation);
        auto layer = violation_polygon->get_layer()->get_name();
        auto polygon = violation_polygon->getPolygon();
        int spacing = violation_polygon->getSpacing();
        int victim_width = violation_polygon->getVictimWidth();
        ieda_solver::GeometryRect bounding_box;
        boost::polygon::extents(bounding_box, polygon);
        auto* layout = engine_manager->get_layout(layer);
        ieda_solver::BgRect bg_rect
            = ieda_solver::BgRect(ieda_solver::BgPoint(boost::polygon::xl(bounding_box), boost::polygon::yl(bounding_box)),
                                  ieda_solver::BgPoint(boost::polygon::xh(bounding_box), boost::polygon::yh(bounding_box)));

        std::vector<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>> tmp = layout->querySubLayouts(
            bg_rect.min_corner().x(), bg_rect.min_corner().y(), bg_rect.max_corner().x(), bg_rect.max_corner().y());
        auto sub_layout_1 = tmp.at(0).second;
        ieda_solver::BgRect bg_rect2
            = ieda_solver::BgRect(ieda_solver::BgPoint(bg_rect.min_corner().x() - spacing, bg_rect.min_corner().y() - spacing),
                                  ieda_solver::BgPoint(bg_rect.max_corner().x() + spacing, bg_rect.max_corner().y() + spacing));
        std::vector<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>> interact_res = layout->querySubLayouts(
            bg_rect2.min_corner().x(), bg_rect2.min_corner().y(), bg_rect2.max_corner().x(), bg_rect2.max_corner().y());
        for (auto& [wire_box, sub_layout_2] : interact_res) {
          if (sub_layout_1->get_id() == sub_layout_2->get_id()) {
            continue;
          }
          // polygon
          auto wire_rect = ieda_solver::GeometryRect(wire_box.min_corner().x(), wire_box.min_corner().y(), wire_box.max_corner().x(),
                                                     wire_box.max_corner().y());
          int wire_width = std::min(ieda_solver::getWireWidth(wire_rect, ieda_solver::K_HORIZONTAL),
                                    ieda_solver::getWireWidth(wire_rect, ieda_solver::K_VERTICAL));
          if (wire_width >= victim_width) {
            std::vector<ieda_solver::GeometryRect> rect_list;
            gtl::get_max_rectangles(rect_list, polygon);
            ieda_solver::GeometryRect closest_intersect;
            ieda_solver::GeometryRect closest_poly_rect;
            int min_dis = INT_MAX;
            for (auto& poly_rect : rect_list) {
              ieda_solver::GeometryRect ge_intersect = poly_rect;
              gtl::generalized_intersect(ge_intersect, wire_rect);
              auto distX = gtl::euclidean_distance(poly_rect, wire_rect, ieda_solver::K_HORIZONTAL);
              auto distY = gtl::euclidean_distance(poly_rect, wire_rect, ieda_solver::K_VERTICAL);
              if (distX && distY && distX + distY < min_dis) {
                min_dis = distX + distY;
                closest_intersect = ge_intersect;
                closest_poly_rect = poly_rect;
              }
            }
            auto distX = gtl::euclidean_distance(closest_poly_rect, wire_rect, ieda_solver::K_HORIZONTAL);
            auto distY = gtl::euclidean_distance(closest_poly_rect, wire_rect, ieda_solver::K_VERTICAL);
            if (distX && distY && min_dis <= 1.2 * spacing) {
              DrcViolationRect* violation_rect = new DrcViolationRect(
                  violation_polygon->get_layer(), violation_polygon->get_violation_type(), gtl::xl(closest_intersect),
                  gtl::yl(closest_intersect), gtl::xh(closest_intersect), gtl::yh(closest_intersect));
              std::set<int> tmp_net_ids;
              tmp_net_ids.insert(sub_layout_1->get_id());
              tmp_net_ids.insert(sub_layout_2->get_id());
              violation_rect->set_net_ids(tmp_net_ids);
              new_violation_list.push_back(violation_rect);
            }
          }
        }

      } else {
        auto* violation_rect = static_cast<DrcViolationRect*>(violation);
        if (violation_rect->get_net_ids().size() <= 0) {
          auto layer = violation_rect->get_layer()->get_name();
          auto* layout = engine_manager->get_layout(layer, violation_rect->get_layer()->is_cut() ? LayoutType::kCut : LayoutType::kRouting);
          if (layout != nullptr) {
            /// if rect is line, enlarge line as a rect to make rtree interact
            int llx = violation_rect->get_llx();
            int lly = violation_rect->get_lly();
            int urx = violation_rect->get_urx();
            int ury = violation_rect->get_ury();
            if (llx == urx) {
              llx -= 2;
              urx += 2;
            }
            if (lly == ury) {
              lly -= 2;
              ury += 2;
            }
            auto net_ids = layout->querySubLayoutNetId(llx, lly, urx, ury);
            while (net_ids.size() > 2) {
              auto it = net_ids.end();
              --it;
              net_ids.erase(it);
            }
            violation_rect->set_net_ids(net_ids);

            // DEBUGOUTPUT(DEBUGHIGHLIGHT("net_ids:\t") << net_ids.size());
          }
        }
        new_violation_list.push_back(violation);
      }
    }
    for (auto* violation : new_violation_list) {
      for (auto it = violation->get_net_ids().begin(); it != violation->get_net_ids().end();) {
        // DEBUGOUTPUT(DEBUGHIGHLIGHT("net_id:\t") << *it);
        if (*it <= NET_ID_OBS) {
          violation->get_inst_ids().insert(NET_ID_OBS - *it);
          it = violation->get_net_ids().erase(it);
        } else {
          ++it;
        }
      }
    }
    violation_list = new_violation_list;
    DEBUGOUTPUT(DEBUGHIGHLIGHT("rule_name:\t") << rule_name << ("\tsize:\t") << violation_list.size());

    DEBUGOUTPUT(DEBUGHIGHLIGHT("rule_name:\t") << rule_name << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
  }
}

std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcViolationManager::get_violation_map(DrcEngineManager* engine_manager)
{
  set_net_ids(engine_manager);
  refineViolation();
  return std::move(_violation_list);
}

std::vector<DrcViolation*>& DrcViolationManager::get_violation_list(ViolationEnumType type)
{
  if (false == _violation_list.contains(type)) {
    _violation_list[type] = std::vector<DrcViolation*>{};
  }
  return _violation_list[type];
}

void DrcViolationManager::addViolation(int llx, int lly, int urx, int ury, ViolationEnumType type, std::set<int> net_id,
                                       std::string layer_name)
{
  idb::IdbLayer* layer = DrcTechRuleInst->findLayer(layer_name);
  DrcViolationRect* violation_rect = new DrcViolationRect(layer, type, llx, lly, urx, ury);
  violation_rect->set_net_ids(net_id);
  auto& violation_list = get_violation_list(type);
  violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
}

void DrcViolationManager::addViolation(ieda_solver::GeometryPolygon polygon, ViolationEnumType type, std::string layer_name, int spacing,
                                       int width)
{
  idb::IdbLayer* layer = DrcTechRuleInst->findLayer(layer_name);
  DrcViolationPolygon* violation_rect = new DrcViolationPolygon(layer, type, spacing, width);
  violation_rect->setPolygon(polygon);
  auto& violation_list = get_violation_list(type);
  violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
}

void DrcViolationManager::refineViolation()
{
  for (auto& [type, violation_list] : _violation_list) {
    violation_list.erase(
        std::remove_if(violation_list.begin(), violation_list.end(), [](DrcViolation* violation) { return violation->ignored(); }),
        violation_list.end());
  }
}

}  // namespace idrc