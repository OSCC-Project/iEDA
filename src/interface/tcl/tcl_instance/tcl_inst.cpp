#include "tcl_inst.h"

#include "Str.hh"
#include "idm.h"
#include "tool_manager.h"

namespace tcl {

TclFpPlaceInst::TclFpPlaceInst(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* inst_name = new TclStringOption("-inst_name", 0, nullptr);
  auto* llx = new TclIntOption("-llx", 0);
  auto* lly = new TclIntOption("-lly", 0);
  auto* orient = new TclStringOption("-orient", 0, nullptr);
  auto* cellmaster = new TclStringOption("-cellmaster", 0, nullptr);
  auto* source = new TclStringOption("-source", 0, nullptr);
  addOption(inst_name);
  addOption(llx);
  addOption(lly);
  addOption(orient);
  addOption(cellmaster);
  addOption(source);
}

unsigned TclFpPlaceInst::check()
{
  TclOption* inst_name = getOptionOrArg("-inst_name");
  TclOption* llxv = getOptionOrArg("-llx");
  TclOption* llyv = getOptionOrArg("-lly");
  TclOption* orientv = getOptionOrArg("-orient");
  TclOption* cellmasterv = getOptionOrArg("-cellmaster");
  TclOption* sourcev = getOptionOrArg("-source");
  LOG_FATAL_IF(!inst_name);
  LOG_FATAL_IF(!llxv);
  LOG_FATAL_IF(!llyv);
  LOG_FATAL_IF(!orientv);
  LOG_FATAL_IF(!cellmasterv);
  LOG_FATAL_IF(!sourcev);
  return 1;
}

unsigned TclFpPlaceInst::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* inst_name = getOptionOrArg("-inst_name");
  TclOption* llxv = getOptionOrArg("-llx");
  TclOption* llyv = getOptionOrArg("-lly");
  TclOption* orientv = getOptionOrArg("-orient");
  TclOption* cellmasterv = getOptionOrArg("-cellmaster");
  TclOption* sourcev = getOptionOrArg("-source");

  auto instance_name = inst_name->getStringVal();
  auto llx = llxv->getIntVal();
  auto lly = llyv->getIntVal();
  auto orient = orientv->getStringVal();
  auto cellmaster = cellmasterv->getStringVal();
  auto source = sourcev->getStringVal();

  string source_str = "";
  if (source != nullptr) {
    source_str = source;
  }
  dmInst->placeInst(instance_name, llx, lly, orient, cellmaster, source_str);

  return 1;
}

}  // namespace tcl
