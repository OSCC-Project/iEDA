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
#include "RVCluster.hpp"
#include "RVComParam.hpp"

namespace idrc {

class RVModel
{
 public:
  RVModel() = default;
  ~RVModel() = default;
  // getter
  std::vector<DRCShape>& get_drc_env_shape_list() { return _drc_env_shape_list; }
  std::vector<DRCShape>& get_drc_result_shape_list() { return _drc_result_shape_list; }
  std::set<ViolationType>& get_drc_check_type_set() { return _drc_check_type_set; }
  std::vector<DRCShape>& get_drc_check_region_list() { return _drc_check_region_list; }
  RVComParam& get_rv_com_param() { return _rv_com_param; }
  std::vector<RVCluster>& get_rv_cluster_list() { return _rv_cluster_list; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  // setter
  void set_drc_env_shape_list(const std::vector<DRCShape>& drc_env_shape_list) { _drc_env_shape_list = drc_env_shape_list; }
  void set_drc_result_shape_list(const std::vector<DRCShape>& drc_result_shape_list) { _drc_result_shape_list = drc_result_shape_list; }
  void set_drc_check_type_set(const std::set<ViolationType>& drc_check_type_set) { _drc_check_type_set = drc_check_type_set; }
  void set_drc_check_region_list(const std::vector<DRCShape>& drc_check_region_list) { _drc_check_region_list = drc_check_region_list; }
  void set_rv_com_param(const RVComParam& rv_com_param) { _rv_com_param = rv_com_param; }
  void set_rv_cluster_list(const std::vector<RVCluster>& rv_cluster_list) { _rv_cluster_list = rv_cluster_list; }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }

 private:
  std::vector<DRCShape> _drc_env_shape_list;
  std::vector<DRCShape> _drc_result_shape_list;
  std::set<ViolationType> _drc_check_type_set;
  std::vector<DRCShape> _drc_check_region_list;
  RVComParam _rv_com_param;
  std::vector<RVCluster> _rv_cluster_list;
  std::vector<Violation> _violation_list;
};

}  // namespace idrc
