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

#include <any>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../database/interaction/ids.hpp"

#if 1  // 前向声明

namespace idb {
class IdbLayerShape;
class IdbRegularWireSegment;
}  // namespace idb

namespace idrc {
enum class ViolationType;
class Violation;
}  // namespace idrc

#endif

namespace idrc {

#define DRCI (idrc::DRCInterface::getInst())

class DRCInterface
{
 public:
  static DRCInterface& getInst();
  static void destroyInst();

#if 1  // 外部调用DRC的API

#if 1  // iDRC
  void initDRC();
  void checkDef();
  std::vector<ids::Violation> getViolationList(std::vector<idb::IdbLayerShape*>& idb_env_shape_list,
                                               std::map<int32_t, std::vector<idb::IdbLayerShape*>>& idb_net_pin_shape_map,
                                               std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& idb_net_result_map);
  void destroyDRC();
#endif

#endif

#if 1  // DRC调用外部的API

#endif

 private:
  static DRCInterface* _drc_interface_instance;

  DRCInterface() = default;
  DRCInterface(const DRCInterface& other) = delete;
  DRCInterface(DRCInterface&& other) = delete;
  ~DRCInterface() = default;
  DRCInterface& operator=(const DRCInterface& other) = delete;
  DRCInterface& operator=(DRCInterface&& other) = delete;
  // function
};

}  // namespace idrc
