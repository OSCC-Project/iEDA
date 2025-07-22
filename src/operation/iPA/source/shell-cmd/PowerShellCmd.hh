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
 * @file PowerShellCmd.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief the classes of ipower shell cmd.
 * @version 0.1
 * @date 2023-05-04
 */

#pragma once

#include "api/Power.hh"
#include "tcl/ScriptEngine.hh"

namespace ipower {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclDoubleOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

/**
 * @brief set the design workspace.
 *
 */
class CmdSetPwrDesignWorkSpace : public TclCmd {
 public:
  explicit CmdSetPwrDesignWorkSpace(const char* cmd_name);
  ~CmdSetPwrDesignWorkSpace() override = default;

  unsigned check();
  unsigned exec();
};

/**
 * @brief The class of read in a VCD file.
 *
 */
class CmdReadVcd : public TclCmd {
 public:
  explicit CmdReadVcd(const char* cmd_name);
  ~CmdReadVcd() override = default;

  unsigned check() override;
  unsigned exec() override;
};

/**
 * @brief read_pg_spef cmd.
 *
 */
class CmdReadPGSpef : public TclCmd {
 public:
  explicit CmdReadPGSpef(const char* cmd_name);
  ~CmdReadPGSpef() override = default;

  unsigned check() override;
  unsigned exec() override;
};

/**
 * @brief report_power command reports power.
 *
 */
class CmdReportPower : public TclCmd {
 public:
  explicit CmdReportPower(const char* cmd_name);
  ~CmdReportPower() override = default;

  unsigned check() override;
  unsigned exec() override;
};

/**
 * @brief report_ir_drop cmd.
 *
 */
class CmdReportIRDrop : public TclCmd {
 public:
  explicit CmdReportIRDrop(const char* cmd_name);
  ~CmdReportIRDrop() override = default;

  unsigned check() override;
  unsigned exec() override;
};

}  // namespace ipower
