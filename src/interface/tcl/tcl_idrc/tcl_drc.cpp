#include "tcl_drc.h"

#include "tool_manager.h"

namespace tcl {

CmdDRCAutoRun::CmdDRCAutoRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_name_option);
  addOption(file_path_option);
}

unsigned CmdDRCAutoRun::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!file_name_option);
  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCAutoRun::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->autoRunDRC(data_config, data_path)) {
    std::cout << "iDRC run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDRCSaveDetailFile::CmdDRCSaveDetailFile(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_path_option);
}

unsigned CmdDRCSaveDetailFile::check()
{
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCSaveDetailFile::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->saveDrcDetailToFile(data_path)) {
    std::cout << "iDRC save detail drc to file success. path = " << data_path << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDRCReadDetailFile::CmdDRCReadDetailFile(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_path_option);
}

unsigned CmdDRCReadDetailFile::check()
{
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);

  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCReadDetailFile::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->readDrcDetailFromFile(data_path)) {
    std::cout << "iDRC read detail file successfully." << std::endl;
  }

  return 1;
}

}  // namespace tcl
