#include "tcl_db.h"

#include "idm.h"
#include "report_manager.h"
#include "tool_manager.h"

namespace tcl {

static const char* const INST_OPT = "-inst";
static const char* const NET_OPT = "-net";
CmdIdbGet::CmdIdbGet(const char* name) : TclCmd(name)
{
  static const char* empty_str = "";
  for (const char* arg : {TCL_PATH, INST_OPT, NET_OPT}) {
    addOption(new TclStringOption(arg, 1, empty_str));
  }
}
unsigned CmdIdbGet::check()
{
  return 1;
}
/**
 * @brief idb_get -path ./result/test/test.rpt -inst xxx
 *        idb_get -path ./result/test/test.rpt -net xxx
 *
 * @return unsigned
 */
unsigned CmdIdbGet::exec()
{
  std::string path = getOptionOrArg(TCL_PATH)->getStringVal();
  std::string inst_name = getOptionOrArg(INST_OPT)->getStringVal();

  if (not inst_name.empty()) {
    rptInst->reportInstance(path, inst_name);
  } else if (std::string net_name = getOptionOrArg(NET_OPT)->getStringVal(); !net_name.empty()) {
    rptInst->reportNet(path, net_name);
  }
  return 1;
}

}  // namespace tcl
