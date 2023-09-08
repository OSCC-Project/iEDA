#include "LocalLegalization.hh"
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

/**
 * @file LocalLegalization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include "CtsDBWrapper.h"
namespace icts {
LocalLegalization::LocalLegalization(Inst* inst, const std::vector<Pin*>& load_pins)
{
  _variable_locations.push_back(inst->get_location());
  std::ranges::for_each(load_pins, [&](Pin* pin) {
    if (pin->isBufferPin()) {
      _variable_locations.push_back(pin->get_location());
    } else {
      _fixed_locations.push_back(pin->get_location());
    }
  });
  legalize();
  inst->set_location(_variable_locations.front());
  if (_variable_locations.size() > 1) {
    std::list<Point> loc_list(_variable_locations.begin() + 1, _variable_locations.end());
    std::ranges::for_each(load_pins, [&](Pin* pin) {
      if (pin->isBufferPin()) {
        pin->set_location(loc_list.front());
        loc_list.pop_front();
      }
    });
  }
}
LocalLegalization::LocalLegalization(const std::vector<Pin*>& pins)
{
  std::ranges::for_each(pins, [&](Pin* pin) {
    if (pin->isBufferPin()) {
      _variable_locations.push_back(pin->get_location());
    } else {
      _fixed_locations.push_back(pin->get_location());
    }
  });
  if (_variable_locations.empty()) {
    return;
  }
  legalize();
  std::list<Point> loc_list(_variable_locations.begin(), _variable_locations.end());
  std::ranges::for_each(pins, [&](Pin* pin) {
    if (pin->isBufferPin()) {
      pin->set_location(loc_list.front());
      loc_list.pop_front();
    }
  });
}
LocalLegalization::LocalLegalization(std::vector<Point>& variable_locations, const std::vector<Point>& fixed_locations)
{
  _variable_locations = variable_locations;
  _fixed_locations = fixed_locations;
  legalize();
  variable_locations = _variable_locations;
}
void LocalLegalization::legalize()
{
  if (_variable_locations.empty()) {
    return;
  }
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  std::set<Point> set;
  std::ranges::for_each(_fixed_locations, [&](const Point& loc) {
    if (set.contains(loc)) {
      LOG_FATAL << "Fixed locations are not legal" << std::endl;
    }
    set.insert(loc);
  });
  std::ranges::for_each(_variable_locations, [&](Point& loc) {
    if (!set.contains(loc)) {
      set.insert(loc);
      return;
    }
    // legalizing
    auto derection = {Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1)};
    bool legal = false;
    int step = 1;
    int max_step = _variable_locations.size() + _fixed_locations.size() + 1;
    while (!legal && step < max_step) {
      for (auto dir : derection) {
        auto new_loc = loc + dir * step;
        if (!set.contains(new_loc) && db_wrapper->withinCore(new_loc)) {
          loc = new_loc;
          set.insert(loc);
          legal = true;
          break;
        }
      }
      ++step;
    }
    if (!legal) {
      LOG_FATAL << "Can not legalize the location" << std::endl;
    }
  });
}
}  // namespace icts