#include "idm.h"
#include "tcl_drc.h"

namespace tcl {

TclDrcCheckNet::TclDrcCheckNet(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(file_name_option);
}

unsigned TclDrcCheckNet::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!file_name_option);

  return 1;
}

unsigned TclDrcCheckNet::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_NAME);
  if (option != nullptr) {
    std::string net_name = option->getStringVal();
    dmInst->isNetConnected(net_name);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclDrcCheckAllNet::TclDrcCheckAllNet(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned TclDrcCheckAllNet::check()
{
  return 1;
}

unsigned TclDrcCheckAllNet::exec()
{
  if (!check()) {
    return 0;
  }

  auto result = dmInst->isAllNetConnected();

  return 1;
}
}  // namespace tcl
