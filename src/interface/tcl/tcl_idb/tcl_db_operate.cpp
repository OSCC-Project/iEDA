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
#include "tcl_db_operate.h"

#include "IdbEnum.h"
#include "IdbNet.h"
#include "idm.h"
#include "tool_manager.h"

namespace tcl {

CmdIdbSetNet::CmdIdbSetNet(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_net_name = new TclStringOption("-net_name", 0);
  auto* tcl_net_type = new TclStringOption("-type", 0);
  addOption(tcl_net_name);
  addOption(tcl_net_type);
}

unsigned CmdIdbSetNet::check()
{
  TclOption* tcl_net_name = getOptionOrArg("-net_name");
  TclOption* tcl_net_type = getOptionOrArg("-type");

  LOG_FATAL_IF(!tcl_net_name);
  LOG_FATAL_IF(!tcl_net_type);
  return 1;
}

unsigned CmdIdbSetNet::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tcl_net_name = getOptionOrArg("-net_name");
  TclOption* tcl_net_type = getOptionOrArg("-type");
  auto net_name = tcl_net_name->getStringVal();
  auto type = tcl_net_type->getStringVal();

  dmInst->setNetType(net_name, type);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdIdbMergeNets::CmdIdbMergeNets(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned CmdIdbMergeNets::check()
{
  return 1;
}

unsigned CmdIdbMergeNets::exec()
{
  if (!check()) {
    return 0;
  }

  dmInst->mergeNets();

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @Brief : clear blockage by type
 * type : 1 placement ; 2 routing; 3 all
 * @param  cmd_name
 */
CmdIdbClearBlockage::CmdIdbClearBlockage(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_type = new TclStringOption("-type", 0);

  addOption(tcl_type);
}

unsigned CmdIdbClearBlockage::check()
{
  TclOption* tcl_type = getOptionOrArg("-type");

  LOG_FATAL_IF(!tcl_type);
  return 1;
}

unsigned CmdIdbClearBlockage::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* tcl_type = getOptionOrArg("-type");
  auto type = tcl_type->getStringVal();

  dmInst->clearBlockage(type);

  return 1;
}

CmdIdbClearBlockageExceptPgNet::CmdIdbClearBlockageExceptPgNet(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned CmdIdbClearBlockageExceptPgNet::check()
{
  return 1;
}

unsigned CmdIdbClearBlockageExceptPgNet::exec()
{
  if (!check()) {
    return 0;
  }

  dmInst->removeBlockageExceptPGNet();

  return 1;
}

static const char* const INST_NAME_OPT = "-inst_name";
CmdIdbDeleteInstance::CmdIdbDeleteInstance(const char* name) : TclCmd(name)
{
  addOption(new TclStringOption(INST_NAME_OPT, 1, EMPTY_STR));
}

unsigned CmdIdbDeleteInstance::check()
{
  return 1;
}

unsigned CmdIdbDeleteInstance::exec()
{
  std::string name = getOptionOrArg(INST_NAME_OPT)->getStringVal();
  bool deleted = dmInst->get_idb_design()->get_instance_list()->remove_instance(name);
  if (deleted) {
    std::cout << "Instance " << name << " removed." << std::endl;
  } else {
    std::cout << "Remove instance" << name << " failed." << std::endl;
  }
  return 1;
}

static const char* const NET_NAME_OPT = "-net_name";
CmdIdbDeleteNet::CmdIdbDeleteNet(const char* name) : TclCmd(name)
{
  addOption(new TclStringOption(NET_NAME_OPT, 1, EMPTY_STR));
}

unsigned CmdIdbDeleteNet::check()
{
  return 1;
}
unsigned CmdIdbDeleteNet::exec()
{
  std::string name = getOptionOrArg(NET_NAME_OPT)->getStringVal();
  bool deleted = dmInst->get_idb_design()->get_net_list()->remove_net(name);
  if (deleted) {
    std::cout << "Net " << name << " removed." << std::endl;
  } else {
    std::cout << "Remove net " << name << " failed." << std::endl;
  }

  return 1;
}

static const char* const CELL_MASTER_OPT = "-cellmaster";
static const char* const COORD_X = "-coord_x";
static const char* const COORD_Y = "-coord_y";
static const char* const ORIENT_OPT = "-orient";
static const char* const TYPE_OPT = "-type";
static const char* const STATUS_OPT = "-status";
/**
 * @brief create_inst -inst_name inst_123 -cellmaster xxx -coord_x 1 -coord_y 1 -orient N -type NETLIST -status UNPLACED}
 *
 * @param name
 */
CmdIdbCreateInstance::CmdIdbCreateInstance(const char* name) : TclCmd(name)
{
  // string options
  for (const auto* opt : {INST_NAME_OPT, CELL_MASTER_OPT, ORIENT_OPT, TYPE_OPT, STATUS_OPT}) {
    addOption(new TclStringOption(opt, 1, EMPTY_STR));
  }
  // int options
  addOption(new ieda::TclIntOption(COORD_X, 1, 0));
  addOption(new ieda::TclIntOption(COORD_Y, 1, 0));
}
unsigned CmdIdbCreateInstance::check()
{
  for (const auto* opt : {INST_NAME_OPT, CELL_MASTER_OPT}) {
    if (char* val = getOptionOrArg(opt)->getStringVal(); val == nullptr || val[0] == '\0') {
      std::cout << "should specify option " << opt << std::endl;
      return 0;
    }
  }

  return 1;
}
unsigned CmdIdbCreateInstance::exec()
{
  if (!check()) {
    return 0;
  }
  std::string inst_name = getOptionOrArg(INST_NAME_OPT)->getStringVal();
  std::string cell_master_name = getOptionOrArg(CELL_MASTER_OPT)->getStringVal();
  int coord_x = getOptionOrArg(COORD_X)->getIntVal();
  int coord_y = getOptionOrArg(COORD_Y)->getIntVal();
  std::string orient_str = getOptionOrArg(ORIENT_OPT)->getStringVal();
  std::string type_str = getOptionOrArg(TYPE_OPT)->getStringVal();
  std::string status_str = getOptionOrArg(STATUS_OPT)->getStringVal();

  auto* instEnum = IdbEnum::GetInstance()->get_instance_property();
  auto* siteEnum = IdbEnum::GetInstance()->get_site_property();

  IdbInstanceType type = instEnum->get_type(type_str);
  IdbOrient orient = siteEnum->get_orient_value(orient_str);
  IdbPlacementStatus status
      = status_str.empty() ? IdbPlacementStatus::kUnplaced : IdbEnum::GetInstance()->get_instance_property()->get_status(status_str);

  idb::IdbInstance* inst = dmInst->createInstance(inst_name, cell_master_name, coord_x, coord_y, orient, type, status);
  if (inst) {
    std::cout << "Create instance " << inst_name << " succeed." << std::endl;
  } else {
    std::cout << "Create instance" << inst_name << " failed." << std::endl;
  }
  return 1;
}

CmdIdbCreateNet::CmdIdbCreateNet(const char* name) : TclCmd(name)
{
  addOption(new TclStringOption(NET_NAME_OPT, 1, EMPTY_STR));
  addOption(new TclStringOption(TYPE_OPT, 1, nullptr));
}
unsigned CmdIdbCreateNet::check()
{
  std::string name = getOptionOrArg(NET_NAME_OPT)->getStringVal();
  if (name.empty()) {
    std::cout << "Create failed, please specify a net name.";
    return 0;
  }
  return 1;
}
unsigned CmdIdbCreateNet::exec()
{
  if (!check()) {
    return 0;
  }
  std::string name = getOptionOrArg(NET_NAME_OPT)->getStringVal();
  std::string type = getOptionOrArg(TYPE_OPT)->getStringVal();
  IdbConnectType conn_type = IdbEnum::GetInstance()->get_connect_property()->get_type(type);

  idb::IdbNet* net = dmInst->createNet(name, conn_type);
  std::cout << "Create net " << name << (net ? " succeed." : " failed.") << std::endl;

  return 1;
}
}  // namespace tcl
