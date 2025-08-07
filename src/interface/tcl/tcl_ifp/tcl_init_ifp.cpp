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
#include "Str.hh"
#include "idm.h"
#include "ifp_api.h"
#include "tcl_ifp.h"
#include "tool_manager.h"
#include <cmath>
namespace tcl {

TclFpInit::TclFpInit(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* core_util = new TclDoubleOption("-core_util", 0);
  auto* cell_area = new TclDoubleOption("-cell_area", 0);
  auto* x_margin = new TclDoubleOption("-x_margin", 0, 10.0);
  auto* y_margin = new TclDoubleOption("-y_margin", 0, 10.0);
  auto* xy_ratio = new TclDoubleOption("-xy_ratio", 0, 1.0);
  auto* tcl_die_area = new TclStringOption("-die_area", 0, nullptr);
  auto* tcl_core_area = new TclStringOption("-core_area", 0, nullptr);
  auto* tcl_core_site = new TclStringOption("-core_site", 0, nullptr);
  auto* tcl_io_site = new TclStringOption("-io_site", 0, nullptr);
  auto* tcl_corner_site = new TclStringOption("-corner_site", 0, nullptr);
  addOption(core_util);
  addOption(cell_area);
  addOption(x_margin);
  addOption(y_margin);
  addOption(xy_ratio);
  addOption(tcl_die_area);
  addOption(tcl_core_area);
  addOption(tcl_core_site);
  addOption(tcl_io_site);
  addOption(tcl_corner_site);
}

unsigned TclFpInit::check()
{
  TclOption* tcl_die_area = getOptionOrArg("-die_area");
  TclOption* tcl_core_area = getOptionOrArg("-core_area");
  TclOption* tcl_core_site = getOptionOrArg("-core_site");
  TclOption* core_util = getOptionOrArg("-core_util");
  //   TclOption* tcl_io_site = getOptionOrArg("-io_site");
  //   TclOption* tcl_corner_site = getOptionOrArg("-corner_site");

  // Check if either explicit areas are provided OR automatic calculation parameters are provided
  bool has_explicit_areas = tcl_die_area && tcl_die_area->is_set_val() &&
                            tcl_core_area && tcl_core_area->is_set_val();
  bool has_auto_calc_params = core_util && core_util->is_set_val();
  // Note: cell_area is now optional - if not provided, it will be retrieved from netlist

  LOG_FATAL_IF(!has_explicit_areas && !has_auto_calc_params)
    << "Either provide explicit -die_area and -core_area, or provide -core_util for automatic calculation (cell_area is optional)";

  LOG_FATAL_IF(!tcl_core_site);
  //   LOG_FATAL_IF(!tcl_io_site);
  //   LOG_FATAL_IF(!tcl_corner_site);
  return 1;
}

unsigned TclFpInit::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_die_area = getOptionOrArg("-die_area");
  TclOption* tcl_core_area = getOptionOrArg("-core_area");
  TclOption* tcl_core_site = getOptionOrArg("-core_site");
  TclOption* tcl_io_site = getOptionOrArg("-io_site");
  TclOption* tcl_corner_site = getOptionOrArg("-corner_site");
  TclOption* core_util = getOptionOrArg("-core_util");
  TclOption* cell_area = getOptionOrArg("-cell_area");
  TclOption* x_margin = getOptionOrArg("-x_margin");
  TclOption* y_margin = getOptionOrArg("-y_margin");
  TclOption* xy_ratio = getOptionOrArg("-xy_ratio");

  ieda::Str str = ieda::Str();

  std::vector<double> die_area;
  std::vector<double> core_area;

  // Check if explicit areas are provided
  bool has_explicit_areas = tcl_die_area && tcl_die_area->is_set_val() &&
                            tcl_core_area && tcl_core_area->is_set_val();

  if (has_explicit_areas) {
    // Use explicitly provided areas
    die_area = str.splitDouble(tcl_die_area->getStringVal(), " ");
    core_area = str.splitDouble(tcl_core_area->getStringVal(), " ");
  } else {
    // Calculate die and core bounding box using core_util
    double util = core_util->getDoubleVal();

    // Get cell area - either from user input or get from iDB
    double cell_area_val;
    if (cell_area && cell_area->is_set_val()) {
      cell_area_val = cell_area->getDoubleVal();
    } else {
      cell_area_val = dmInst->instanceArea(IdbInstanceType::kMax);
    }

    double x_margin_val = x_margin->getDoubleVal();
    double y_margin_val = y_margin->getDoubleVal();
    double ratio = xy_ratio->getDoubleVal();

    // Calculate core area based on cell area and utilization
    double total_core_area = cell_area_val / util;

    // Calculate core dimensions based on aspect ratio
    double core_height = sqrt(total_core_area / ratio);
    double core_width = total_core_area / core_height;

    // Calculate die dimensions by adding margins
    double die_width = core_width + 2 * x_margin_val;
    double die_height = core_height + 2 * y_margin_val;

    // Set die area (llx, lly, urx, ury)
    die_area = {0.0, 0.0, die_width, die_height};

    // Set core area with margins
    core_area = {x_margin_val, y_margin_val, x_margin_val + core_width, y_margin_val + core_height};
  }

  string core_site = tcl_core_site->getStringVal();
  string io_site = tcl_io_site->getStringVal() == nullptr ? core_site : tcl_io_site->getStringVal();
  string corner_site = tcl_corner_site->getStringVal() == nullptr ? io_site : tcl_corner_site->getStringVal();

  fpApiInst->initDie(die_area[0], die_area[1], die_area[2], die_area[3]);
  fpApiInst->initCore(core_area[0], core_area[1], core_area[2], core_area[3], core_site, io_site, corner_site);
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclFpMakeTracks::TclFpMakeTracks(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_layer = new TclStringOption("-layer", 0);
  auto* tcl_x_start = new TclIntOption("-x_start", 0);
  auto* tcl_x_step = new TclIntOption("-x_step", 0);
  auto* tcl_y_start = new TclIntOption("-y_start", 0);
  auto* tcl_y_step = new TclIntOption("-y_step", 0);
  addOption(tcl_layer);
  addOption(tcl_x_start);
  addOption(tcl_x_step);
  addOption(tcl_y_start);
  addOption(tcl_y_step);
}

unsigned TclFpMakeTracks::check()
{
  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_x_start = getOptionOrArg("-x_start");
  TclOption* tcl_x_step = getOptionOrArg("-x_step");
  TclOption* tcl_y_start = getOptionOrArg("-y_start");
  TclOption* tcl_y_step = getOptionOrArg("-y_step");

  LOG_FATAL_IF(!tcl_layer);
  LOG_FATAL_IF(!tcl_x_start);
  LOG_FATAL_IF(!tcl_x_step);
  LOG_FATAL_IF(!tcl_y_start);
  LOG_FATAL_IF(!tcl_y_step);
  return 1;
}

unsigned TclFpMakeTracks::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_x_start = getOptionOrArg("-x_start");
  TclOption* tcl_x_step = getOptionOrArg("-x_step");
  TclOption* tcl_y_start = getOptionOrArg("-y_start");
  TclOption* tcl_y_step = getOptionOrArg("-y_step");

  auto layer = tcl_layer->getStringVal();
  auto x_start = tcl_x_start->getIntVal();
  auto x_step = tcl_x_step->getIntVal();
  auto y_start = tcl_y_start->getIntVal();
  auto y_step = tcl_y_step->getIntVal();

  fpApiInst->makeTracks(layer, x_start, x_step, y_start, y_step);

  return 1;
}

}  // namespace tcl
