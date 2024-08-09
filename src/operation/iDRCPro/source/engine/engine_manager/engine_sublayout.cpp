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
#include "engine_sublayout.h"

#include "engine_geometry_creator.h"

namespace idrc {

DrcEngineSubLayout::DrcEngineSubLayout(int id)
{
  _id = id;

  ieda_solver::EngineGeometryCreator geo_creator;
  _engine = geo_creator.create();
}

DrcEngineSubLayout::~DrcEngineSubLayout()
{
  if (_engine != nullptr) {
    delete _engine;
    _engine = nullptr;
  }

  _check_nets.clear();
}

bool DrcEngineSubLayout::isIntersect(int llx, int lly, int urx, int ury)
{
  return _engine->isIntersect(llx, lly, urx, ury);
}

void DrcEngineSubLayout::markChecked(int net_id)
{
  _check_nets.insert(net_id);
}

bool DrcEngineSubLayout::hasChecked(int net_id)
{
  return _check_nets.find(net_id) != _check_nets.end() ? true : false;
}

bool DrcEngineSubLayout::clearChecked(){
    _check_nets.clear();
}

}  // namespace idrc