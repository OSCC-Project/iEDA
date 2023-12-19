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
#include "engine_geometry.h"

namespace idrc {
class DrcEngineSubLayout
{
 public:
  DrcEngineSubLayout(int id);
  ~DrcEngineSubLayout();

  int get_id() { return _id; }
  ieda_solver::EngineGeometry* get_engine() { return _engine; }

 private:
  /**
   * _id : net id
   * _engine : a geometry ptr including all shapes in one net
   */
  int _id = -1;
  ieda_solver::EngineGeometry* _engine = nullptr;
};

}  // namespace idrc