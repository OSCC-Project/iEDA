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
#include "engine_layout.h"

namespace idrc {
DrcEngineLayout::~DrcEngineLayout()
{
  for (auto& [id, sub_layout] : _sub_layouts) {
    if (sub_layout != nullptr) {
      delete sub_layout;
      sub_layout = nullptr;
    }
  }

  _sub_layouts.clear();
}

bool DrcEngineLayout::addRect(int llx, int lly, int urx, int ury, int net_id)
{
  auto* engine = get_net_engine(net_id);
  if (engine == nullptr) {
    return false;
  }

  engine->addRect(llx, lly, urx, ury);

  return true;
}

DrcEngineSubLayout* DrcEngineLayout::get_sub_layout(int net_id)
{
  auto* sub_layout = _sub_layouts[net_id];
  if (sub_layout == nullptr) {
    sub_layout = new DrcEngineSubLayout(net_id);
    _sub_layouts[net_id] = sub_layout;
  }

  return sub_layout;
}

ieda_solver::EngineGeometry* DrcEngineLayout::get_net_engine(int net_id)
{
  auto* sub_layout = get_sub_layout(net_id);

  return sub_layout == nullptr ? nullptr : sub_layout->get_engine();
}

}  // namespace idrc