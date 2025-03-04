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

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "DRCViolationType.h"
#include "idrc_violation.h"
#include "json/json.hpp"

namespace idb {
class IdbRegularWireSegment;
class IdbLayerShape;
}  // namespace idb

namespace idrc {
class DrcManager;

class DrcApi
{
 public:
  DrcApi() {}
  ~DrcApi() = default;
  void init(std::string config = "");
  void exit();

  std::map<ViolationEnumType, std::vector<DrcViolation*>> check(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                                std::map<int, std::vector<idb::IdbLayerShape*>>& pin_data,
                                                                std::map<int, std::vector<idb::IdbRegularWireSegment*>>& routing_data,
                                                                std::set<ViolationEnumType> check_select = {});
  std::map<ViolationEnumType, std::vector<DrcViolation*>> checkByBox(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                                     std::map<int, std::vector<idb::IdbLayerShape*>>& pin_data,
                                                                     std::map<int, std::vector<idb::IdbRegularWireSegment*>>& routing_data,
                                                                     std::set<ViolationEnumType> check_select = {});
  std::map<ViolationEnumType, std::vector<DrcViolation*>> checkDef();

 private:
};

}  // namespace idrc