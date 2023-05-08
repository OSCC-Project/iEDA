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

  addOption(tcl_layer);
  addOption(tcl_width);
  addOption(tcl_height);
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

  auto layer = tcl_layer->getStringVal();
  auto width = tcl_width->getIntVal();
  auto height = tcl_height->getIntVal();

  fpApiInst->autoPlacePins(layer, width, height);

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
  auto* orient = new TclStringOption("-edge", 0);
  auto* begin = new TclDoubleOption("-begin_pos", 0, 0);
  auto* end = new TclDoubleOption("-end_pos", 0, 0);
  auto* source = new TclStringOption("-source", 0, nullptr);
  addOption(source);
  addOption(fillers_name);
  addOption(prefix);
  addOption(orient);
  addOption(begin);
  addOption(end);
}

unsigned TclFpPlaceIOFiller::check()
{
  TclOption* fillers_name = getOptionOrArg("-filler_types");
  TclOption* prefix = getOptionOrArg("-prefix");
  TclOption* orient = getOptionOrArg("-edge");
  TclOption* begin = getOptionOrArg("-begin_pos");
  TclOption* end = getOptionOrArg("-end_pos");
  TclOption* sourcev = getOptionOrArg("-source");
  LOG_FATAL_IF(!sourcev);
  LOG_FATAL_IF(!fillers_name);
  LOG_FATAL_IF(!prefix);
  LOG_FATAL_IF(!orient);
  LOG_FATAL_IF(!begin);
  LOG_FATAL_IF(!end);
  return 1;
}

unsigned TclFpPlaceIOFiller::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* fillers_name = getOptionOrArg("-filler_types");
  TclOption* prefix = getOptionOrArg("-prefix");
  TclOption* orient = getOptionOrArg("-edge");
  TclOption* begin = getOptionOrArg("-begin_pos");
  TclOption* end = getOptionOrArg("-end_pos");
  TclOption* sourcev = getOptionOrArg("-source");

  string pre, ori, source_str;
  double beg = 0, en = 0;

  auto fill = fillers_name->getStringList();
  auto source = sourcev->getStringVal();
  if (source != nullptr) {
    source_str = source;
  } else {
    source_str = "";
  }

  if (prefix->is_set_val() == 0) {
    pre = "IOFill";
  } else {
    pre = prefix->getStringVal();
  }

  if (begin != nullptr) {
    beg = begin->getDoubleVal();
  }

  if (end != nullptr) {
    en = end->getDoubleVal();
  }

  if (orient != nullptr && orient->is_set_val() != 0) {
    ori = orient->getStringVal();
  }

  fpApiInst->placeIOFiller(fill, pre, ori, beg, en, source_str);

  return 1;
}

}  // namespace tcl
