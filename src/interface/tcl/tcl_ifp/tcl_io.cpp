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

namespace tcl {

TclFpPlacePins::TclFpPlacePins(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_layer = new TclStringOption("-layer", 0);
  auto* tcl_width = new TclIntOption("-width", 0);
  auto* tcl_height = new TclIntOption("-height", 0);
  auto* tcl_sides = new TclStringListOption("-sides", 1);  /// -sides "left right top bottom"

  addOption(tcl_layer);
  addOption(tcl_width);
  addOption(tcl_height);
  addOption(tcl_sides);
}

unsigned TclFpPlacePins::check()
{
  return 1;
}

unsigned TclFpPlacePins::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_layer = getOptionOrArg("-layer");
  TclOption* tcl_width = getOptionOrArg("-width");
  TclOption* tcl_height = getOptionOrArg("-height");
  TclOption* tcl_sides = getOptionOrArg("-sides");

  auto layer = tcl_layer->getStringVal();
  auto width = tcl_width->getIntVal();
  auto height = tcl_height->getIntVal();
  vector<string> sides = {};

  if (tcl_sides != nullptr) {
    sides = tcl_sides->getStringList();
  }

  fpApiInst->autoPlacePins(layer, width, height, sides);

  std::cout << "Floorplan place pins." << std::endl;
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TclFpPlacePort::TclFpPlacePort(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* pin_name = new TclStringOption("-pin_name", 0);
  auto* offset_x = new TclIntOption("-offset_x", 0);
  auto* offset_y = new TclIntOption("-offset_y", 0);
  auto* width = new TclIntOption("-width", 0);
  auto* height = new TclIntOption("-height", 0);
  auto* layer_name = new TclStringOption("-layer", 0);
  addOption(pin_name);
  addOption(offset_x);
  addOption(offset_y);
  addOption(width);
  addOption(height);
  addOption(layer_name);
}

unsigned TclFpPlacePort::check()
{
  TclOption* pin_name = getOptionOrArg("-pin_name");
  TclOption* offset_x = getOptionOrArg("-offset_x");
  TclOption* offset_y = getOptionOrArg("-offset_y");
  TclOption* width = getOptionOrArg("-width");
  TclOption* height = getOptionOrArg("-height");
  TclOption* layer_namev = getOptionOrArg("-layer");
  LOG_FATAL_IF(!pin_name);
  LOG_FATAL_IF(!offset_x);
  LOG_FATAL_IF(!offset_y);
  LOG_FATAL_IF(!layer_namev);
  LOG_FATAL_IF(!width);
  LOG_FATAL_IF(!height);
  return 1;
}

unsigned TclFpPlacePort::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* tcl_pin_name = getOptionOrArg("-pin_name");
  TclOption* tcl_offset_x = getOptionOrArg("-offset_x");
  TclOption* tcl_offset_y = getOptionOrArg("-offset_y");
  TclOption* tcl_width = getOptionOrArg("-width");
  TclOption* tcl_height = getOptionOrArg("-height");
  TclOption* tcl_layer_name = getOptionOrArg("-layer");

  auto pin_name = tcl_pin_name->getStringVal();
  auto offset_x = tcl_offset_x->getIntVal();
  auto offset_y = tcl_offset_y->getIntVal();
  auto width = tcl_width->getIntVal();
  auto height = tcl_height->getIntVal();
  auto layer_name = tcl_layer_name->getStringVal();

  fpApiInst->placePort(pin_name, offset_x, offset_y, width, height, layer_name);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclFpPlaceIOFiller::TclFpPlaceIOFiller(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* fillers_name = new TclStringListOption("-filler_types", 0);
  auto* prefix = new TclStringOption("-prefix", 0);

  addOption(fillers_name);
  addOption(prefix);
}

unsigned TclFpPlaceIOFiller::check()
{
  TclOption* fillers_name = getOptionOrArg("-filler_types");
  TclOption* prefix = getOptionOrArg("-prefix");

  LOG_FATAL_IF(!fillers_name);
  LOG_FATAL_IF(!prefix);
  return 1;
}

unsigned TclFpPlaceIOFiller::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* fillers_name = getOptionOrArg("-filler_types");
  TclOption* prefix = getOptionOrArg("-prefix");

  string pre;

  auto fill = fillers_name->getStringList();

  if (prefix->is_set_val() == 0) {
    pre = "IOFill";
  } else {
    pre = prefix->getStringVal();
  }

  fpApiInst->placeIOFiller(fill, pre);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclFpAutoPlaceIO::TclFpAutoPlaceIO(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_pads = new TclStringListOption("-pads", 0);
  auto* tcl_corners = new TclStringListOption("-conners", 0);

  addOption(tcl_pads);
  addOption(tcl_corners);
}

unsigned TclFpAutoPlaceIO::check()
{
  auto* tcl_pads = getOptionOrArg("-pads");
  auto* tcl_corners = getOptionOrArg("-conners");

  //   LOG_FATAL_IF(!tcl_pads);
  //   LOG_FATAL_IF(!tcl_pads);

  return 1;
}

unsigned TclFpAutoPlaceIO::exec()
{
  if (!check()) {
    return 0;
  }

  auto* tcl_pads = getOptionOrArg("-pads");
  auto* tcl_corners = getOptionOrArg("-conners");

  std::vector<std::string> pad_names = tcl_pads != nullptr ? tcl_pads->getStringList() : std::vector<std::string>{};
  std::vector<std::string> corner_names = tcl_corners != nullptr ? tcl_corners->getStringList() : std::vector<std::string>{};

  fpApiInst->autoPlacePad(pad_names, corner_names);

  return 1;
}

}  // namespace tcl
