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

#include "../../../database/interaction/RT_DRC/ids.hpp"

#if 1  // 前向声明

namespace idrc {
class DRCModel;
class DRCBox;
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
  void destroyDRC();
  std::vector<ids::Violation> getViolationList(std::vector<ids::Shape> ids_shape_list);
  DRCModel initDRCModel(std::vector<ids::Shape>& ids_shape_list);
  std::vector<ids::Violation> getViolationList(DRCModel& drc_model);
#endif

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
