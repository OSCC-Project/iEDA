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
#include "DRCEngine.hpp"

#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void DRCEngine::initInst()
{
  if (_de_instance == nullptr) {
    _de_instance = new DRCEngine();
  }
}

DRCEngine& DRCEngine::getInst()
{
  if (_de_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_de_instance;
}

void DRCEngine::destroyInst()
{
  if (_de_instance != nullptr) {
    delete _de_instance;
    _de_instance = nullptr;
  }
}

// function

void DRCEngine::init()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  RTI.initIDRC();
  buildIgnoredViolationSet();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> DRCEngine::getViolationList(DETask& de_task)
{
  getViolationListByInterface(de_task);
  filterViolationList(de_task);
  checkViolationList(de_task);
  if (de_task.get_proc_type() == DEProcType::kGet) {
    buildViolationList(de_task);
  }
  return de_task.get_violation_list();
}

void DRCEngine::addTempIgnoredViolation(std::vector<Violation>& violation_list)
{
  for (Violation& violation : violation_list) {
    _temp_ignored_violation_set.insert(violation);
  }
}

void DRCEngine::clearTempIgnoredViolationSet()
{
  _temp_ignored_violation_set.clear();
}

void DRCEngine::destroy()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  RTI.destroyIDRC();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

void DRCEngine::buildIgnoredViolationSet()
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("ignore_violation");
    std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
    std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
    for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
      for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
        for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
          if (net_idx == -1) {
            for (auto& fixed_rect : fixed_rect_set) {
              env_shape_list.emplace_back(fixed_rect, is_routing);
            }
          } else {
            for (auto& fixed_rect : fixed_rect_set) {
              net_pin_shape_map[net_idx].emplace_back(fixed_rect, is_routing);
            }
          }
        }
      }
    }
    std::set<int32_t> need_checked_net_set;
    for (Net& net : net_list) {
      need_checked_net_set.insert(net.get_net_idx());
    }
    de_task.set_proc_type(DEProcType::kIgnore);
    de_task.set_net_type(DENetType::kRouteHybrid);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  for (Violation violation : getViolationList(de_task)) {
    _ignored_violation_set.insert(violation);
  }
}

void DRCEngine::getViolationListByInterface(DETask& de_task)
{
  de_task.set_violation_list(RTI.getViolationList(de_task.get_env_shape_list(), de_task.get_net_pin_shape_map(), de_task.get_net_result_map(),
                                                  de_task.get_net_patch_map(), de_task.get_check_type_set(), de_task.get_check_region_list()));
}

void DRCEngine::filterViolationList(DETask& de_task)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : de_task.get_violation_list()) {
    if (violation.get_violation_type() == ViolationType::kNone) {
      // 未知规则舍弃
      continue;
    }
    if (skipViolation(de_task, violation)) {
      // 跳过的类型舍弃
      continue;
    }
    bool exist_checked_net = false;
    {
      for (int32_t violation_net_idx : violation.get_violation_net_set()) {
        if (RTUTIL.exist(de_task.get_need_checked_net_set(), violation_net_idx)) {
          exist_checked_net = true;
          break;
        }
      }
    }
    if (!exist_checked_net) {
      // net不包含布线net的舍弃
      continue;
    }
    if (RTUTIL.exist(_ignored_violation_set, violation) || RTUTIL.exist(_temp_ignored_violation_set, violation)) {
      // 自带的违例舍弃
      continue;
    }
    new_violation_list.push_back(violation);
  }
  de_task.set_violation_list(new_violation_list);
}

void DRCEngine::checkViolationList(DETask& de_task)
{
  for (Violation& violation : de_task.get_violation_list()) {
    if (!violation.get_is_routing()) {
      RTLOG.error(Loc::current(), "The violations in the cut layer!");
    }
    if (violation.get_violation_net_set().size() > 2) {
      RTLOG.error(Loc::current(), "The violation_net_set size > 2!");
    }
  }
}

void DRCEngine::buildViolationList(DETask& de_task)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  std::vector<Violation> new_violation_list;
  for (Violation& violation : de_task.get_violation_list()) {
    for (Violation new_violation : getExpandedViolationList(de_task, violation)) {
      EXTLayerRect& violation_shape = new_violation.get_violation_shape();
      violation_shape.set_grid_rect(RTUTIL.getClosedGCellGridRect(violation_shape.get_real_rect(), gcell_axis));
      new_violation_list.push_back(new_violation);
    }
  }
  de_task.set_violation_list(new_violation_list);
}

#if 1  // aux

bool DRCEngine::skipViolation(DETask& de_task, Violation& violation)
{
  std::vector<Violation> expanded_violation_list = getExpandedViolationList(de_task, violation);
  return expanded_violation_list.empty();
}

std::vector<Violation> DRCEngine::getExpandedViolationList(DETask& de_task, Violation& violation)
{
  DENetType& net_type = de_task.get_net_type();
  if (net_type == DENetType::kNone) {
    RTLOG.error(Loc::current(), "The net_type is none!");
  }
  PlanarRect new_real_rect = violation.get_violation_shape().get_real_rect();
  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  if (net_type == DENetType::kRouteHybrid) {
    switch (violation.get_violation_type()) {
      case ViolationType::kAdjacentCutSpacing:
        break;
      case ViolationType::kCornerFillSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kCornerSpacing:
        break;
      case ViolationType::kCutEOLSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {0, +1});
        break;
      case ViolationType::kCutShort:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0, +1});
        break;
      case ViolationType::kDifferentLayerCutSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {0, +1, +2});
        break;
      case ViolationType::kEnclosure:
        break;
      case ViolationType::kEnclosureEdge:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kEnclosureParallel:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kEndOfLineSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kFloatingPatch:
        break;
      case ViolationType::kJogToJogSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kMaximumWidth:
        break;
      case ViolationType::kMaxViaStack:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0, +1});
        break;
      case ViolationType::kMetalShort:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kMinHole:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kMinimumArea:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kMinimumCut:
        break;
      case ViolationType::kMinimumWidth:
        break;
      case ViolationType::kMinStep:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kNonsufficientMetalOverlap:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kNotchSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kOffGridOrWrongWay:
        break;
      case ViolationType::kOutOfDie:
        break;
      case ViolationType::kParallelRunLengthSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {-1, 0, +1});
        break;
      case ViolationType::kSameLayerCutSpacing:
        new_real_rect = enlargeRect(new_real_rect, violation.get_required_size());
        layer_routing_list = expandLayer(violation, {0, +1});
        break;
      default:
        RTLOG.error(Loc::current(), "No violation type!");
        break;
    }
  } else if (net_type == DENetType::kPatchHybrid) {
    switch (violation.get_violation_type()) {
      case ViolationType::kAdjacentCutSpacing:
        break;
      case ViolationType::kCornerFillSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kCornerSpacing:
        break;
      case ViolationType::kCutEOLSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kCutShort:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kDifferentLayerCutSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kEnclosure:
        break;
      case ViolationType::kEnclosureEdge:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kEnclosureParallel:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kEndOfLineSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kFloatingPatch:
        break;
      case ViolationType::kJogToJogSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMaximumWidth:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMaxViaStack:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMetalShort:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMinHole:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMinimumArea:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMinimumCut:
        break;
      case ViolationType::kMinimumWidth:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kMinStep:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kNonsufficientMetalOverlap:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kNotchSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kOffGridOrWrongWay:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kOutOfDie:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kParallelRunLengthSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      case ViolationType::kSameLayerCutSpacing:
        new_real_rect = enlargeRect(new_real_rect, 0);
        layer_routing_list = expandLayer(violation, {0});
        break;
      default:
        RTLOG.error(Loc::current(), "No violation type!");
        break;
    }
  }
  std::vector<Violation> expanded_violation_list;
  for (std::pair<int32_t, bool>& layer_routing : layer_routing_list) {
    Violation expanded_violation = violation;
    expanded_violation.get_violation_shape().set_real_rect(new_real_rect);
    expanded_violation.get_violation_shape().set_layer_idx(layer_routing.first);
    expanded_violation.set_is_routing(layer_routing.second);
    expanded_violation_list.push_back(expanded_violation);
  }
  return expanded_violation_list;
}

PlanarRect DRCEngine::enlargeRect(PlanarRect& real_rect, int32_t required_size)
{
  int32_t enlarged_x_size = 0;
  if (real_rect.getXSpan() < required_size) {
    enlarged_x_size = required_size - real_rect.getXSpan();
  }
  int32_t enlarged_y_size = 0;
  if (real_rect.getYSpan() < required_size) {
    enlarged_y_size = required_size - real_rect.getYSpan();
  }
  return RTUTIL.getEnlargedRect(real_rect, enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
}

std::vector<std::pair<int32_t, bool>> DRCEngine::expandLayer(Violation& violation, std::vector<int32_t> offset_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t violation_layer_idx = violation.get_violation_shape().get_layer_idx();

  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  for (int32_t offset : offset_list) {
    int32_t offset_layer_idx = violation_layer_idx + offset;
    if (0 <= offset_layer_idx && offset_layer_idx <= (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
      layer_routing_list.emplace_back(offset_layer_idx, true);
    }
  }
  return layer_routing_list;
}

#endif

}  // namespace irt
