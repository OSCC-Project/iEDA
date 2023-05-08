#include "Str.hh"
#include "idm.h"
#include "ifp_api.h"
#include "tcl_ifp.h"
#include "tool_manager.h"
namespace tcl {

TclFpInit::TclFpInit(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_file_name = new TclStringOption("-die_area", 0, nullptr);
  auto* tcl_core_area = new TclStringOption("-core_area", 0, nullptr);
  auto* tcl_core_site = new TclStringOption("-core_site", 0, nullptr);
  auto* tcl_io_site = new TclStringOption("-io_site", 0, nullptr);
  auto* tcl_corner_site = new TclStringOption("-corner_site", 0, nullptr);
  addOption(tcl_file_name);
  addOption(tcl_core_area);
  addOption(tcl_core_site);
  addOption(tcl_io_site);
  addOption(tcl_corner_site);
}

unsigned TclFpInit::check()
{
  TclOption* tcl_file_name = getOptionOrArg("-die_area");
  TclOption* tcl_core_area = getOptionOrArg("-core_area");
  TclOption* tcl_core_site = getOptionOrArg("-core_site");
  //   TclOption* tcl_io_site = getOptionOrArg("-io_site");
  //   TclOption* tcl_corner_site = getOptionOrArg("-corner_site");
  LOG_FATAL_IF(!tcl_file_name);
  LOG_FATAL_IF(!tcl_core_area);
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

  TclOption* tcl_file_name = getOptionOrArg("-die_area");
  TclOption* tcl_core_area = getOptionOrArg("-core_area");
  TclOption* tcl_core_site = getOptionOrArg("-core_site");
  TclOption* tcl_io_site = getOptionOrArg("-io_site");
  TclOption* tcl_corner_site = getOptionOrArg("-corner_site");

  ieda::Str str = ieda::Str();

  // auto idb_design = dmInst->get_idb_design();

  std::vector<double> die_area = str.splitDouble(tcl_file_name->getStringVal(), " ");
  std::vector<double> core_area = str.splitDouble(tcl_core_area->getStringVal(), " ");
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
