#include "tcl_report_route.h"

namespace tcl {

static const auto TCL_NET = "-net";
static const auto TCL_SUMMARY = "-summary";

CmdReportRoute::CmdReportRoute(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(TCL_PATH, 1, EMPTY_STR));
  addOption(new TclStringOption(TCL_NET, 1, EMPTY_STR));
  addOption(new ieda::TclIntOption(TCL_SUMMARY, 1, 1));
}
unsigned CmdReportRoute::exec()
{
  auto* file_path = getOptionOrArg(TCL_PATH)->getStringVal();
  auto* net = getOptionOrArg(TCL_NET)->getStringVal();
  auto summary = getOptionOrArg(TCL_SUMMARY)->getIntVal();
  rptInst->reportRoute(file_path, net, summary);
  return 1;
};

}  // namespace tcl