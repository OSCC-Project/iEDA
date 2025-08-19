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
#include "py_ifp.h"

#include <idm.h>
#include <ifp_api.h>
#include <tool_manager.h>

#include <Str.hh>

namespace python_interface {

bool fpInit(const std::string& die_area, const std::string& core_area, const std::string& core_site, const std::string& io_site,
            const std::string& corner_site, double core_util, double x_margin, double y_margin, double xy_ratio, double cell_area)
{
  std::vector<double> die = ieda::Str::splitDouble(die_area.c_str(), " ");
  std::vector<double> core = ieda::Str::splitDouble(core_area.c_str(), " ");
  if (die.empty() || core.empty()) {
    // Get cell area - either from user input or get from iDB
    if (cell_area <= 0) {
      cell_area = dmInst->instanceArea(IdbInstanceType::kMax);
    }

    // Calculate core area based on cell area and utilization
    double total_core_area = cell_area / core_util;

    // Calculate core dimensions based on aspect ratio
    double core_height = sqrt(total_core_area / xy_ratio);
    double core_width = total_core_area / core_height;

    // Calculate die dimensions by adding margins
    double die_width = core_width + 2 * x_margin;
    double die_height = core_height + 2 * y_margin;

    // Set die area (llx, lly, urx, ury)
    die = {0.0, 0.0, die_width, die_height};

    // Set core area with margins
    core = {x_margin, y_margin, x_margin + core_width, y_margin + core_height};
  }

  fpApiInst->initDie(die[0], die[1], die[2], die[3]);
  fpApiInst->initCore(core[0], core[1], core[2], core[3], core_site, io_site, corner_site);
  return true;
}

bool fpMakeTracks(const std::string& layer, int x_start, int x_step, int y_start, int y_step)
{
  bool make_ok = fpApiInst->makeTracks(layer, x_start, x_step, y_start, y_step);
  return make_ok;
}

bool fpPlacePins(const std::string& layer, int width, int height, std::vector<std::string>& sides)
{
  bool place_ok = fpApiInst->autoPlacePins(layer, width, height, sides);
  return place_ok;
}

bool fpPlacePort(const std::string& pin_name, int offset_x, int offset_y, int width, int height, const std::string& layer)
{
  bool place_ok = fpApiInst->placePort(pin_name, offset_x, offset_y, width, height, layer);
  return place_ok;
}

bool fpPlaceIOFiller(std::vector<std::string>& filler_types, const std::string& prefix)
{
  bool place_ok = fpApiInst->placeIOFiller(filler_types, prefix);
  return place_ok;
}

bool fpAddPlacementBlockage(const std::string& box)
{
  auto blk = ieda::Str::splitInt(box.c_str(), " ");
  int32_t llx = blk[0];
  int32_t lly = blk[1];
  int32_t urx = blk[2];
  int32_t ury = blk[3];

  dmInst->addPlacementBlockage(llx, lly, urx, ury);
  return true;
}

bool fpAddPlacementHalo(const std::string& inst_name, const std::string& distance)
{
  auto distance_val = ieda::Str::splitInt(distance.c_str(), " ");
  int32_t left = distance_val[0];
  int32_t bottom = distance_val[1];
  int32_t right = distance_val[2];
  int32_t top = distance_val[3];

  dmInst->addPlacementHalo(inst_name, top, bottom, left, right);
  return true;
}

bool fpAddRoutingBlockage(const std::string& layer, const std::string& box, bool exceptpgnet)
{
  auto layers = ieda::Str::split(layer.c_str(), " ");
  auto box_result = ieda::Str::splitInt(box.c_str(), " ");

  int32_t llx = box_result[0];
  int32_t lly = box_result[1];
  int32_t urx = box_result[2];
  int32_t ury = box_result[3];

  dmInst->addRoutingBlockage(llx, lly, urx, ury, layers, exceptpgnet);
  return true;
}

bool fpAddRoutingHalo(const std::string& layer, const std::string& distance, bool exceptpgnet, const std::string& inst_name)
{
  auto layers = ieda::Str::split(layer.c_str(), " ");
  auto box_result = ieda::Str::splitInt(distance.c_str(), " ");
  int32_t left = box_result[0];
  int32_t bottom = box_result[1];
  int32_t right = box_result[2];
  int32_t top = box_result[3];

  dmInst->addRoutingHalo(inst_name, layers, top, bottom, left, right, exceptpgnet);
  return true;
}

bool fpTapCell(const std::string& tapcell, double distance, const std::string& endcap)
{
  return fpApiInst->tapCells(distance, tapcell, endcap);
}
}  // namespace python_interface