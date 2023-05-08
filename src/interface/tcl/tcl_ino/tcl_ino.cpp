#include "tcl_ino.h"

#include "tool_manager.h"

namespace tcl {

CmdNORunFixFanout::CmdNORunFixFanout(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdNORunFixFanout::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdNORunFixFanout::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  if (iplf::tmInst->RunNOFixFanout(data_config)) {
    std::cout << "iNO fixfanout run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace tcl
