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
#include "LayerAssigner.hpp"

#include "GDSPlotter.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void LayerAssigner::initInst()
{
  if (_la_instance == nullptr) {
    _la_instance = new LayerAssigner();
  }
}

LayerAssigner& LayerAssigner::getInst()
{
  if (_la_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_la_instance;
}

void LayerAssigner::destroyInst()
{
  if (_la_instance != nullptr) {
    delete _la_instance;
    _la_instance = nullptr;
  }
}

// function

void LayerAssigner::assign()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

LayerAssigner* LayerAssigner::_la_instance = nullptr;

}  // namespace irt
