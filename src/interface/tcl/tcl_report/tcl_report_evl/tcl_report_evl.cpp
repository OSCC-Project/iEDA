#include "tcl_report_evl.h"
#include "report_manager.h"

namespace tcl {

CmdReportWL::CmdReportWL(const char* cmd) : TclCmd(cmd)
{
  addOption(new TclStringOption(TCL_PATH, 1));
}
unsigned CmdReportWL::check()
{
  return 1;
}

CMD_CLASS_DEFAULT_EXEC(CmdReportWL,
                       rptInst->reportWL(getOptionOrArg(TCL_PATH)->getStringVal() ? getOptionOrArg(TCL_PATH)->getStringVal() : "");)

CmdReportCong::CmdReportCong(const char* cmd) : TclCmd(cmd)
{
  addOption(new TclStringOption(TCL_PATH, 1));
}
unsigned CmdReportCong::check()
{
  return 1;
}
CMD_CLASS_DEFAULT_EXEC(CmdReportCong,
                       rptInst->reportCongestion(getOptionOrArg(TCL_PATH)->getStringVal() ? getOptionOrArg(TCL_PATH)->getStringVal() : ""));

}  // namespace tcl