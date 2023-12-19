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

#include "idrc.h"
#include "idrc_config.h"
#include "idrc_data.h"
#include "idrc_dm.h"
#include "idrc_rule_stratagy.h"
#include "rule_builder.h"

namespace idrc {

void DrcApi::init(std::string config)
{
  DrcConfigInst->init(config);

  /// tech rule is a singleton pattern, so it must be inited if drc starts
  DrcRuleBuilder builder;
  builder.build();
}

void DrcApi::exit()
{
  DrcTechRuleInst->destroyInst();
}

/**
 * drc_shape_list在region_query的环境里产生的违例信息，如spacing
 * 关注于非同net之间的违例
 */
std::map<std::string, std::vector<BaseViolationInfo>> DrcApi::getEnvViolationInfo(BaseRegion& base_region,
                                                                                  const std::vector<DRCCheckType>& check_type_list,
                                                                                  std::vector<BaseShape>& drc_shape_list)
{
  DrcManager drc_manager;
  drc_manager.init();

  auto* data_manager = drc_manager.get_data_manager();
  auto* rule_manager = drc_manager.get_rule_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr || rule_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  data_manager->set_region(&base_region);
  data_manager->set_target_shapes(&drc_shape_list);
  /// set drc rule stratagy by rt paramter
  /// tbd
  rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  drc_manager.engineStart(DrcCheckerType::kRT);

  drc_manager.buildCondition();

  drc_manager.check();

  return violation_manager->get_rt_violation_map();
}
/**
 * drc_shape_list组成的自身违例信息，如min_area,min_step
 * 关注于net内的违例
 */
std::map<std::string, std::vector<BaseViolationInfo>> DrcApi::getSelfViolationInfo(const std::vector<DRCCheckType>& check_type_list,
                                                                                   std::vector<BaseShape>& drc_shape_list)
{
  DrcManager drc_manager;
  drc_manager.init();

  auto* data_manager = drc_manager.get_data_manager();
  auto* rule_manager = drc_manager.get_rule_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr || rule_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  data_manager->set_target_shapes(&drc_shape_list);
  /// set drc rule stratagy by rt paramter
  /// tbd
  rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  drc_manager.engineStart(DrcCheckerType::kRT);

  drc_manager.buildCondition();

  drc_manager.check();

  return violation_manager->get_rt_violation_map();
}
/**
 * check DRC violation for DEF file
 * initialize data from idb
 */
std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcApi::checkDef()
{
  DrcManager drc_manager;

  auto* data_manager = drc_manager.get_data_manager();
  auto* rule_manager = drc_manager.get_rule_manager();
  auto* violation_manager = drc_manager.get_violation_manager();
  if (data_manager == nullptr || rule_manager == nullptr || violation_manager == nullptr) {
    return {};
  }

  /// set drc rule stratagy by rt paramter
  /// tbd
  rule_manager->get_stratagy()->set_stratagy_type(DrcStratagyType::kCheckFast);

  drc_manager.engineStart(DrcCheckerType::kDef);

  drc_manager.buildCondition();

  drc_manager.check();

  return violation_manager->get_violation_map();
}

}  // namespace idrc