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

#include "idrc_api.h"

#include "idm.h"
#include "idrc.h"
#include "idrc_config.h"
#include "idrc_data.h"
#include "idrc_dm.h"
#include "tech_rules.h"
#ifdef USE_PROFILER
#include <gperftools/profiler.h>
#endif

namespace idrc {

void DrcApi::init(std::string config)
{
  DrcConfigInst->init(config);

  /// tech rule is a singleton pattern, so it must be inited if drc starts
  DrcTechRuleInst->init();
}

void DrcApi::exit()
{
  DrcTechRuleInst->destroyInst();
}

struct DRCBox
{
  // input
  std::vector<idb::IdbLayerShape*> env_shape_list;
  std::map<int, std::vector<idb::IdbLayerShape*>> pin_data;
  std::map<int, std::vector<idb::IdbRegularWireSegment*>> routing_data;
  // data
  int32_t real_ll_x;
  int32_t real_ll_y;
  int32_t real_ur_x;
  int32_t real_ur_y;
  // output
  std::map<ViolationEnumType, std::vector<DrcViolation*>> type_violation_list_map;
};

std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::check(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                                      std::map<int, std::vector<idb::IdbLayerShape*>>& pin_data,
                                                                      std::map<int, std::vector<idb::IdbRegularWireSegment*>>& routing_data,
                                                                      std::set<ViolationEnumType> check_select)
{
  int32_t shape_num = 0;
  shape_num += env_shape_list.size();
  for (auto& [net_idx, pin_shape_list] : pin_data) {
    shape_num += pin_shape_list.size();
  }
  for (auto& [net_idx, routing_segment_list] : routing_data) {
    shape_num += routing_segment_list.size();
  }
  if (shape_num > 20000) {
    int32_t box_size = 10000;
    int32_t expand_size = 500;

    // 初始化 设计边界 grid个数 box列表
    int32_t design_ll_x = INT32_MAX;
    int32_t design_ll_y = INT32_MAX;
    int32_t design_ur_x = INT32_MIN;
    int32_t design_ur_y = INT32_MIN;
    int32_t design_x_size = -1;
    int32_t design_y_size = -1;
    std::vector<DRCBox> drc_box_list;
    {
      for (idb::IdbLayerShape* env_shape : env_shape_list) {
        for (idb::IdbRect* rect : env_shape->get_rect_list()) {
          design_ll_x = std::min(design_ll_x, rect->get_low_x());
          design_ll_y = std::min(design_ll_y, rect->get_low_y());
          design_ur_x = std::max(design_ur_x, rect->get_high_x());
          design_ur_y = std::max(design_ur_y, rect->get_high_y());
        }
      }
      for (auto& [net_idx, pin_shape_list] : pin_data) {
        for (idb::IdbLayerShape* pin_shape : pin_shape_list) {
          for (idb::IdbRect* rect : pin_shape->get_rect_list()) {
            design_ll_x = std::min(design_ll_x, rect->get_low_x());
            design_ll_y = std::min(design_ll_y, rect->get_low_y());
            design_ur_x = std::max(design_ur_x, rect->get_high_x());
            design_ur_y = std::max(design_ur_y, rect->get_high_y());
          }
        }
      }
      for (auto& [net_idx, routing_segment_list] : routing_data) {
        for (idb::IdbRegularWireSegment* routing_segment : routing_segment_list) {
          if (routing_segment->is_wire()) {
            if (routing_segment->get_point_number() != 2) {
              std::cout << "idrc : Unknown wire!" << std::endl;
            }
            int32_t low_x = routing_segment->get_point_start()->get_x();
            int32_t low_y = routing_segment->get_point_start()->get_y();
            int32_t high_x = routing_segment->get_point_end()->get_x();
            int32_t high_y = routing_segment->get_point_end()->get_y();
            if (high_x < low_x) {
              std::swap(low_x, high_x);
            }
            if (high_y < low_y) {
              std::swap(low_y, high_y);
            }
            design_ll_x = std::min(design_ll_x, low_x);
            design_ll_y = std::min(design_ll_y, low_y);
            design_ur_x = std::max(design_ur_x, high_x);
            design_ur_y = std::max(design_ur_y, high_y);
          } else if (routing_segment->is_via()) {
            int32_t point_x = routing_segment->get_point_start()->get_x();
            int32_t point_y = routing_segment->get_point_start()->get_y();
            design_ll_x = std::min(design_ll_x, point_x);
            design_ll_y = std::min(design_ll_y, point_y);
            design_ur_x = std::max(design_ur_x, point_x);
            design_ur_y = std::max(design_ur_y, point_y);
          } else if (routing_segment->is_rect()) {
            int32_t point_x = routing_segment->get_point_start()->get_x();
            int32_t point_y = routing_segment->get_point_start()->get_y();
            idb::IdbRect* rect = routing_segment->get_delta_rect();
            design_ll_x = std::min(design_ll_x, rect->get_low_x() + point_x);
            design_ll_y = std::min(design_ll_y, rect->get_low_y() + point_y);
            design_ur_x = std::max(design_ur_x, rect->get_high_x() + point_x);
            design_ur_y = std::max(design_ur_y, rect->get_high_y() + point_y);
          } else {
            std::cout << "idrc : Unknown type!" << std::endl;
          }
        }
      }
      // 粗暴的防止越界
      design_ll_x--;
      design_ll_y--;
      design_ur_x++;
      design_ur_y++;
      design_x_size = std::ceil((design_ur_x - design_ll_x) / 1.0 / box_size);
      design_y_size = std::ceil((design_ur_y - design_ll_y) / 1.0 / box_size);
      drc_box_list.resize(design_x_size * design_y_size);
    }
    // 向Box内部添加数据
    {
      for (idb::IdbLayerShape* env_shape : env_shape_list) {
        for (idb::IdbRect* rect : env_shape->get_rect_list()) {
          int32_t ll_x = std::max(design_ll_x, rect->get_low_x() - design_ll_x - expand_size);
          int32_t ll_y = std::max(design_ll_y, rect->get_low_y() - design_ll_y - expand_size);
          int32_t ur_x = std::min(design_ur_x, rect->get_high_x() - design_ll_x + expand_size);
          int32_t ur_y = std::min(design_ur_y, rect->get_high_y() - design_ll_y + expand_size);
          for (int32_t grid_x = (ll_x / box_size); grid_x <= (ur_x / box_size); grid_x++) {
            for (int32_t grid_y = (ll_y / box_size); grid_y <= (ur_y / box_size); grid_y++) {
              drc_box_list[grid_x + grid_y * design_x_size].env_shape_list.push_back(env_shape);
            }
          }
        }
      }
      for (auto& [net_idx, pin_shape_list] : pin_data) {
        for (idb::IdbLayerShape* pin_shape : pin_shape_list) {
          for (idb::IdbRect* rect : pin_shape->get_rect_list()) {
            int32_t ll_x = std::max(design_ll_x, rect->get_low_x() - design_ll_x - expand_size);
            int32_t ll_y = std::max(design_ll_y, rect->get_low_y() - design_ll_y - expand_size);
            int32_t ur_x = std::min(design_ur_x, rect->get_high_x() - design_ll_x + expand_size);
            int32_t ur_y = std::min(design_ur_y, rect->get_high_y() - design_ll_y + expand_size);
            for (int32_t grid_x = (ll_x / box_size); grid_x <= (ur_x / box_size); grid_x++) {
              for (int32_t grid_y = (ll_y / box_size); grid_y <= (ur_y / box_size); grid_y++) {
                drc_box_list[grid_x + grid_y * design_x_size].pin_data[net_idx].push_back(pin_shape);
              }
            }
          }
        }
      }
      for (auto& [net_idx, routing_segment_list] : routing_data) {
        for (idb::IdbRegularWireSegment* routing_segment : routing_segment_list) {
          if (routing_segment->is_wire()) {
            if (routing_segment->get_point_number() != 2) {
              std::cout << "idrc : Unknown wire!" << std::endl;
            }
            int32_t low_x = routing_segment->get_point_start()->get_x();
            int32_t low_y = routing_segment->get_point_start()->get_y();
            int32_t high_x = routing_segment->get_point_end()->get_x();
            int32_t high_y = routing_segment->get_point_end()->get_y();
            if (high_x < low_x) {
              std::swap(low_x, high_x);
            }
            if (high_y < low_y) {
              std::swap(low_y, high_y);
            }
            int32_t ll_x = std::max(design_ll_x, low_x - design_ll_x - expand_size);
            int32_t ll_y = std::max(design_ll_y, low_y - design_ll_y - expand_size);
            int32_t ur_x = std::min(design_ur_x, high_x - design_ll_x + expand_size);
            int32_t ur_y = std::min(design_ur_y, high_y - design_ll_y + expand_size);
            for (int32_t grid_x = (ll_x / box_size); grid_x <= (ur_x / box_size); grid_x++) {
              for (int32_t grid_y = (ll_y / box_size); grid_y <= (ur_y / box_size); grid_y++) {
                drc_box_list[grid_x + grid_y * design_x_size].routing_data[net_idx].push_back(routing_segment);
              }
            }
          } else if (routing_segment->is_via()) {
            int32_t point_x = routing_segment->get_point_start()->get_x();
            int32_t point_y = routing_segment->get_point_start()->get_y();
            int32_t ll_x = std::max(design_ll_x, point_x - design_ll_x - expand_size);
            int32_t ll_y = std::max(design_ll_y, point_y - design_ll_y - expand_size);
            int32_t ur_x = std::min(design_ur_x, point_x - design_ll_x + expand_size);
            int32_t ur_y = std::min(design_ur_y, point_y - design_ll_y + expand_size);
            for (int32_t grid_x = (ll_x / box_size); grid_x <= (ur_x / box_size); grid_x++) {
              for (int32_t grid_y = (ll_y / box_size); grid_y <= (ur_y / box_size); grid_y++) {
                drc_box_list[grid_x + grid_y * design_x_size].routing_data[net_idx].push_back(routing_segment);
              }
            }
          } else if (routing_segment->is_rect()) {
            int32_t point_x = routing_segment->get_point_start()->get_x();
            int32_t point_y = routing_segment->get_point_start()->get_y();
            idb::IdbRect* rect = routing_segment->get_delta_rect();
            int32_t ll_x = std::max(design_ll_x, rect->get_low_x() + point_x - design_ll_x - expand_size);
            int32_t ll_y = std::max(design_ll_y, rect->get_low_y() + point_y - design_ll_y - expand_size);
            int32_t ur_x = std::min(design_ur_x, rect->get_high_x() + point_x - design_ll_x + expand_size);
            int32_t ur_y = std::min(design_ur_y, rect->get_high_y() + point_y - design_ll_y + expand_size);
            for (int32_t grid_x = (ll_x / box_size); grid_x <= (ur_x / box_size); grid_x++) {
              for (int32_t grid_y = (ll_y / box_size); grid_y <= (ur_y / box_size); grid_y++) {
                drc_box_list[grid_x + grid_y * design_x_size].routing_data[net_idx].push_back(routing_segment);
              }
            }
          } else {
            std::cout << "idrc : Unknown type!" << std::endl;
          }
        }
      }
    }
    // 处理Box数据
    {
#pragma omp parallel for
      for (size_t drc_box_idx = 0; drc_box_idx < drc_box_list.size(); drc_box_idx++) {
        DRCBox& drc_box = drc_box_list[drc_box_idx];
        std::vector<idb::IdbLayerShape*>& box_env_shape_list = drc_box.env_shape_list;
        std::map<int, std::vector<idb::IdbLayerShape*>>& box_pin_data = drc_box.pin_data;
        std::map<int, std::vector<idb::IdbRegularWireSegment*>>& box_routing_data = drc_box.routing_data;
        std::map<idrc::ViolationEnumType, std::vector<idrc::DrcViolation*>>& box_type_violation_list_map = drc_box.type_violation_list_map;
        // 去除冗余数据
        {
          std::sort(box_env_shape_list.begin(), box_env_shape_list.end());
          box_env_shape_list.erase(std::unique(box_env_shape_list.begin(), box_env_shape_list.end()), box_env_shape_list.end());
          for (auto& [net_idx, pin_shape_list] : box_pin_data) {
            std::sort(pin_shape_list.begin(), pin_shape_list.end());
            pin_shape_list.erase(std::unique(pin_shape_list.begin(), pin_shape_list.end()), pin_shape_list.end());
          }
          for (auto& [net_idx, routing_segment_list] : box_routing_data) {
            std::sort(routing_segment_list.begin(), routing_segment_list.end());
            routing_segment_list.erase(std::unique(routing_segment_list.begin(), routing_segment_list.end()), routing_segment_list.end());
          }
        }
        // 计算设计违例
        {
          box_type_violation_list_map = checkByBox(box_env_shape_list, box_pin_data, box_routing_data);
        }
        // 清除额外违例
        {
          int32_t grid_ll_x = drc_box_idx % design_x_size;
          int32_t grid_ll_y = drc_box_idx / design_x_size;

          int32_t box_ll_x = grid_ll_x * box_size + design_ll_x;
          int32_t box_ll_y = grid_ll_y * box_size + design_ll_y;
          int32_t box_ur_x = (grid_ll_x + 1) * box_size + design_ll_x;
          int32_t box_ur_y = (grid_ll_y + 1) * box_size + design_ll_y;

          for (auto& [type, violation_list] : box_type_violation_list_map) {
            std::vector<idrc::DrcViolation*> new_violation_list;
            for (idrc::DrcViolation* violation : violation_list) {
              idrc::DrcViolationRect* violation_rect = static_cast<idrc::DrcViolationRect*>(violation);
              int32_t x_spacing = std::max(violation_rect->get_llx() - box_ur_x, box_ll_x - violation_rect->get_urx());
              int32_t y_spacing = std::max(violation_rect->get_lly() - box_ur_y, box_ll_y - violation_rect->get_ury());
              if (x_spacing < 0 && y_spacing < 0) {
                new_violation_list.push_back(violation);
              }
            }
            violation_list = new_violation_list;
          }
        }
      }
    }
    // 得到最后结果
    std::map<ViolationEnumType, std::vector<DrcViolation*>> type_violation_list_map;
    {
      for (DRCBox& drc_box : drc_box_list) {
        for (auto& [type, violation_list] : drc_box.type_violation_list_map) {
          for (DrcViolation* violation : violation_list) {
            type_violation_list_map[type].push_back(violation);
          }
        }
      }
    }
    return type_violation_list_map;
  } else {
    return checkByBox(env_shape_list, pin_data, routing_data);
  }
}

std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::checkByBox(
    std::vector<idb::IdbLayerShape*>& env_shape_list, std::map<int, std::vector<idb::IdbLayerShape*>>& pin_data,
    std::map<int, std::vector<idb::IdbRegularWireSegment*>>& routing_data, std::set<ViolationEnumType> check_select)
{
  // 有bug,对于没有任何形状传入时,需要耗费大量时间
  int32_t shape_num = 0;
  shape_num += env_shape_list.size();
  for (auto& [net_idx, pin_shape_list] : pin_data) {
    shape_num += pin_shape_list.size();
  }
  for (auto& [net_idx, routing_segment_list] : routing_data) {
    shape_num += routing_segment_list.size();
  }
  if (shape_num == 0) {
    return {};
  }

  DrcManager drc_manager;

  auto* data_manager = drc_manager.get_data_manager();
  // auto* rule_manager = drc_manager.get_rule_manager();
  auto* condition_manager = drc_manager.get_condition_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr /*|| rule_manager == nullptr */ || condition_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  condition_manager->set_check_select(check_select);

  /// set drc rule stratagy by rt paramter
  /// tbd
  // rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  data_manager->set_env_shapes(&env_shape_list);
  data_manager->set_pin_data(&pin_data);
  data_manager->set_routing_data(&routing_data);

  auto check_type = (env_shape_list.size() + pin_data.size() + routing_data.size()) > 0 ? DrcCheckerType::kRT : DrcCheckerType::kDef;

  condition_manager->set_check_type(check_type);

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : check def" << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_engine;
#endif
  drc_manager.dataInit(check_type);
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : engine start"
              << " runtime = " << stats_engine.elapsedRunTime() << " memory = " << stats_engine.memoryDelta() << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_build_condition;
#endif
  drc_manager.dataOperate();
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : dataOperate"
              << " runtime = " << stats_build_condition.elapsedRunTime() << " memory = " << stats_build_condition.memoryDelta()
              << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_check;
#endif
  drc_manager.dataCheck();

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : dataCheck"
              << " runtime = " << stats_check.elapsedRunTime() << " memory = " << stats_check.memoryDelta() << std::endl;
  }
#endif

  return violation_manager->get_violation_map(drc_manager.get_engine()->get_engine_manager());
}
/**
 * check DRC violation for DEF file
 * initialize data from idb
 */
std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::checkDef()
{
  std::vector<idb::IdbLayerShape*> env_shape_list;
  std::map<int, std::vector<idb::IdbLayerShape*>> pin_data;
  std::map<int, std::vector<idb::IdbRegularWireSegment*>> routing_data;
  std::set<ViolationEnumType> check_select;

  DrcManager drc_manager;

  auto* data_manager = drc_manager.get_data_manager();
  // auto* rule_manager = drc_manager.get_rule_manager();
  auto* condition_manager = drc_manager.get_condition_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr /*|| rule_manager == nullptr */ || condition_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  condition_manager->set_check_select(check_select);

  /// set drc rule stratagy by rt paramter
  /// tbd
  // rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  data_manager->set_env_shapes(&env_shape_list);
  data_manager->set_pin_data(&pin_data);
  data_manager->set_routing_data(&routing_data);

  // auto check_type = (env_shape_list.size() + pin_data.size() + routing_data.size()) > 0 ? DrcCheckerType::kRT : DrcCheckerType::kDef;
  auto check_type = DrcCheckerType::kDef;

  condition_manager->set_check_type(check_type);

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : check def" << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_engine;
#endif
  drc_manager.dataInit(check_type);
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : engine start"
              << " runtime = " << stats_engine.elapsedRunTime() << " memory = " << stats_engine.memoryDelta() << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_build_condition;
#endif
  drc_manager.dataOperate();
#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : build condition"
              << " runtime = " << stats_build_condition.elapsedRunTime() << " memory = " << stats_build_condition.memoryDelta()
              << std::endl;
  }
#endif

#ifdef DEBUG_IDRC_API
  ieda::Stats stats_check;
#endif
  drc_manager.dataCheck();

#ifdef DEBUG_IDRC_API
  if (DrcCheckerType::kDef == check_type) {
    std::cout << "idrc : check"
              << " runtime = " << stats_check.elapsedRunTime() << " memory = " << stats_check.memoryDelta() << std::endl;
  }
#endif

  return violation_manager->get_violation_map(drc_manager.get_engine()->get_engine_manager());
}

}  // namespace idrc