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

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Monitor.hpp"

namespace irt {

#define RTVR (irt::ViolationRepairer::getInst())

class ViolationRepairer
{
 public:
  static void initInst();
  static ViolationRepairer& getInst();
  static void destroyInst();
  // function
  void repair();

 private:
  // self
  static ViolationRepairer* _vr_instance;

  ViolationRepairer() = default;
  ViolationRepairer(const ViolationRepairer& other) = delete;
  ViolationRepairer(ViolationRepairer&& other) = delete;
  ~ViolationRepairer() = default;
  ViolationRepairer& operator=(const ViolationRepairer& other) = delete;
  ViolationRepairer& operator=(ViolationRepairer&& other) = delete;
  // function
};

}  // namespace irt
