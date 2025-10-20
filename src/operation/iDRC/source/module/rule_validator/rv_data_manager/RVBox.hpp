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

#include "DRCShape.hpp"
#include "LayerRect.hpp"
#include "PlanarRect.hpp"
#include "RVComParam.hpp"
#include "RVSummary.hpp"
#include "Segment.hpp"
#include "Violation.hpp"

namespace idrc {

class RVBox
{
 public:
  RVBox() = default;
  ~RVBox() = default;
  // getter
  int32_t get_box_idx() const { return _box_idx; }
  PlanarRect& get_box_rect() { return _box_rect; }
  RVComParam* get_rv_com_param() { return _rv_com_param; }
  std::vector<DRCShape*>& get_drc_env_shape_list() { return _drc_env_shape_list; }
  std::vector<DRCShape*>& get_drc_result_shape_list() { return _drc_result_shape_list; }
  std::set<ViolationType>* get_drc_check_type_set() { return _drc_check_type_set; }
  std::vector<DRCShape>* get_drc_check_region_list() { return _drc_check_region_list; }
  std::set<Violation, CmpViolation>& get_env_violation_set() { return _env_violation_set; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  std::map<ViolationType, std::set<Violation, CmpViolation>>& get_type_violation_map() { return _type_violation_map; }
  std::map<ViolationType, std::set<Violation, CmpViolation>>& get_type_golden_violation_map() { return _type_golden_violation_map; }
  RVSummary& get_rv_summary() { return _rv_summary; }
  // setter
  void set_box_idx(const int32_t box_idx) { _box_idx = box_idx; }
  void set_box_rect(const PlanarRect& box_rect) { _box_rect = box_rect; }
  void set_rv_com_param(RVComParam* rv_com_param) { _rv_com_param = rv_com_param; }
  void set_drc_env_shape_list(const std::vector<DRCShape*>& drc_env_shape_list) { _drc_env_shape_list = drc_env_shape_list; }
  void set_drc_result_shape_list(const std::vector<DRCShape*>& drc_result_shape_list) { _drc_result_shape_list = drc_result_shape_list; }
  void set_drc_check_type_set(std::set<ViolationType>* drc_check_type_set) { _drc_check_type_set = drc_check_type_set; }
  void set_drc_check_region_list(std::vector<DRCShape>* drc_check_region_list) { _drc_check_region_list = drc_check_region_list; }
  void set_env_violation_set(const std::set<Violation, CmpViolation>& env_violation_set) { _env_violation_set = env_violation_set; }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  void set_type_violation_map(const std::map<ViolationType, std::set<Violation, CmpViolation>>& type_violation_map)
  {
    _type_violation_map = type_violation_map;
  }
  void set_type_golden_violation_map(const std::map<ViolationType, std::set<Violation, CmpViolation>>& type_golden_violation_map)
  {
    _type_golden_violation_map = type_golden_violation_map;
  }
  void set_rv_summary(const RVSummary& rv_summary) { _rv_summary = rv_summary; }
  // function
 private:
  int32_t _box_idx = -1;
  PlanarRect _box_rect;
  RVComParam* _rv_com_param = nullptr;
  std::vector<DRCShape*> _drc_env_shape_list;
  std::vector<DRCShape*> _drc_result_shape_list;
  std::set<ViolationType>* _drc_check_type_set;
  std::vector<DRCShape>* _drc_check_region_list;
  std::set<Violation, CmpViolation> _env_violation_set;
  std::vector<Violation> _violation_list;
  std::map<ViolationType, std::set<Violation, CmpViolation>> _type_violation_map;
  std::map<ViolationType, std::set<Violation, CmpViolation>> _type_golden_violation_map;
  RVSummary _rv_summary;
};

}  // namespace idrc
