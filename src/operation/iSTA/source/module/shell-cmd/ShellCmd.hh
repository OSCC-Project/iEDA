// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file SellCmd.hh
 * @author Wang Hao (harryw0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-09-28
 */
#pragma once

#include "netlist/DesignObject.hh"
#include "sta/Sta.hh"
#include "tcl/ScriptEngine.hh"

namespace ista {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclCmds;
using ieda::TclDoubleListOption;
using ieda::TclDoubleOption;
using ieda::TclEncodeResult;
using ieda::TclIntListOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringListListOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

/**
 * @brief set the design workspace.
 *
 */
class CmdSetDesignWorkSpace : public TclCmd {
 public:
  explicit CmdSetDesignWorkSpace(const char* cmd_name);
  ~CmdSetDesignWorkSpace() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief Reads in one or more (will support) Verilog files.
 *
 */
class CmdReadVerilog : public TclCmd {
 public:
  explicit CmdReadVerilog(const char* cmd_name);
  ~CmdReadVerilog() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief Read lef def file to convert netlist.
 *
 */
class CmdReadLefDef : public TclCmd {
 public:
  explicit CmdReadLefDef(const char* cmd_name);
  ~CmdReadLefDef() override = default;

  unsigned check();
  unsigned exec();
};

class CmdTESTSLL : public TclCmd {
 public:
  explicit CmdTESTSLL(const char* cmd_name);
  ~CmdTESTSLL() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief Reads in one Lib files.
 *
 */
class CmdReadLiberty : public TclCmd {
 public:
  explicit CmdReadLiberty(const char* cmd_name);
  ~CmdReadLiberty() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief Specifies the name of the design to be linked; the default is the
 * current design.
 *
 */
class CmdLinkDesign : public TclCmd {
 public:
  explicit CmdLinkDesign(const char* cmd_name);
  ~CmdLinkDesign() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief read_spef command.
 *
 */
class CmdReadSpef : public TclCmd {
 public:
  explicit CmdReadSpef(const char* cmd_name);
  ~CmdReadSpef() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief Reads in a script in Synopsys Design Constraints (SDC) format.
 *
 */
class CmdReadSdc : public TclCmd {
 public:
  explicit CmdReadSdc(const char* cmd_name);
  ~CmdReadSdc() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief report_checks command reports paths in the design.
 *
 */
class CmdReportTiming : public TclCmd {
 public:
  explicit CmdReportTiming(const char* cmd_name);
  ~CmdReportTiming() override = default;

  unsigned printHelp();
  unsigned check();
  unsigned exec();
};

/**
 * @brief report_check_types command reports the slack for each type of timing
 * and design rule constraint. The keyword options allow a subset of the
 * constraint types to be reported.
 *
 */
class CmdReportConstraint : public TclCmd {
 public:
  explicit CmdReportConstraint(const char* cmd_name);
  ~CmdReportConstraint() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief read def file, change to netlist,finally write verilog file.
 *
 */
class CmdDefToVerilog : public TclCmd {
 public:
  explicit CmdDefToVerilog(const char* cmd_name);
  ~CmdDefToVerilog() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief read verilog file, change to def, finally write def file.
 *
 */
class CmdVerilogToDef : public TclCmd {
 public:
  explicit CmdVerilogToDef(const char* cmd_name);
  ~CmdVerilogToDef() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief dump graph data.
 * 
 */
class CmdDumpGraphData : public TclCmd {
 public:
  explicit CmdDumpGraphData(const char* cmd_name);
  ~CmdDumpGraphData() override = default;

  unsigned check();
  unsigned exec();
};

}  // namespace ista
